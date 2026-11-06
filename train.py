import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
from utils.utils import *
from utils.models import *
from tqdm import tqdm
from torchvision.utils import save_image

def parse_args():
    parser = argparse.ArgumentParser(description="Utility script for various tasks.")

    parser.add_argument(
        "--content_dir",
        type=str,
        default="D:/Brinda/NST_CODE/content-data",
        help="Location of the content directory."
    )
    parser.add_argument(
            "--style_dir",
            type=str,
            default="D:/Brinda/NST_CODE/style-data",
            help="Location of the style directory."
        )
    parser.add_argument(
        "--vgg",
        type=str,
        default="D:/Brinda/NST_CODE/vgg_normalised.pth",
        help="Location of the pre-trained VGG model."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default='experiment_1',
        help="Name of the experiment."
    )
 
    parser.add_argument(
        "--final_size", type=int, default=256, help="Final size of the images.")
    parser.add_argument(
        "--content-size", type=int, default=256, help="Size of the content images.")
    parser.add_argument(
        "--style-size", type=int, default=256, help="Size of the style images.")
    parser.add_argument(
        "--crop", action="store_true", default=True, help="Whether to crop the images.")

    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument(
        "--lr_decay", type=float, default=5e-5, help="Learning rate decay for the optimizer.")

    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of epochs for training.")

    parser.add_argument(
        "--content_weight", type=float, default=1.0, help="Weight for the content loss.")
    parser.add_argument(
        "--style_weight", type=float, default=10.0, help="Weight for the style loss.")
    parser.add_argument(
        "--log_interval", type=int, default=1, help="Interval for logging training progress.")
    parser.add_argument(
        "--save_interval", type=int, default=1, help="Interval for saving model checkpoints.")

    parser.add_argument(
        "--resume", action="store_true", default=False, help="Whether to resume training from a checkpoint.")
    parser.add_argument(
        "--decoder_path", type=str, default="D:/Brinda/NST_CODE/experiment/decoder_epoch_1.pth", help="Path to the decoder checkpoint.")
    parser.add_argument(
        "--optimizer_path", type=str, default=None, help="Path to the optimizer checkpoint.")

    return parser.parse_args()

def main():
    args = parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path("experiment") / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    #Save argument values
    with open(save_dir / "args.txt", "w") as args_file:
        for key, value in vars(args).items():
            args_file.write(f"{key}: {value}\n")

    content_transform = get_transform(args.content_size, args.crop, args.final_size)
    style_transform = get_transform(args.style_size, args.crop, args.final_size)

    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)

    content_dataloader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True,
                                    pin_memory=True, drop_last=False)
    style_dataloader = DataLoader(style_dataset, batch_size=args.batch_size, shuffle=True,
                                  pin_memory=True, drop_last=False)

    print(f"Content dataset size: {len(content_dataset)}")
    print(f"Style dataset size: {len(style_dataset)}")

    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda = lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch)
    )

    if args.resume:
        decoder.load_state_dict(torch.load(args.decoder_path))
        optimizer.load_state_dict(torch.load(args.optimizer_path))

    print("Starting training...")

    mse_loss = torch.nn.MSELoss()

    encoder.eval()
    
    for epoch in range(args.epochs):

        running_loss = 0.0
        running_content_loss = 0.0
        running_style_loss = 0.0
        
        progress_bar = tqdm(zip(content_dataloader, style_dataloader),
                            total=min(len(content_dataloader), len(style_dataloader)))

        for content_batch, style_batch in progress_bar:

            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            c_features = encoder(content_batch)
            s_features = encoder(style_batch)

            t = adaptive_instance_normalization(c_features[-1], s_features[-1])

            g = decoder(t)

            g_features = encoder(g)

            loss_c = mse_loss(g_features[-1], t) * args.content_weight

            loss_s = 0.0

            for g_f, s_f in zip(g_features, s_features):
                g_f_mean, g_f_std = calc_mean_std(g_f)
                s_f_mean, s_f_std = calc_mean_std(s_f)
                loss_s += mse_loss(g_f_mean, s_f_mean) + mse_loss(g_f_std, s_f_std)

            loss_s *= args.style_weight

            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f"Epoch [{epoch + 1}/{args.epochs}] "
                                         f"Loss: {loss.item():.4f} "
                                         f"Content Loss: {loss_c.item():.4f} "
                                         f"Style Loss: {loss_s.item():.4f}")

            running_loss += loss.item()
            running_content_loss += loss_c.item()
            running_style_loss += loss_s.item()

        scheduler.step()

        running_loss /= len(content_dataloader)
        running_content_loss /= len(content_dataloader)
        running_style_loss /= len(content_dataloader)

        if (epoch + 1) % args.log_interval == 0:
            tqdm.write(f"Epoch [{epoch + 1}/{args.epochs}] "
                  f"Loss: {running_loss:.4f} "
                  f"Content Loss: {running_content_loss:.4f} "
                  f"Style Loss: {running_style_loss:.4f}")

        if (epoch + 1) % args.save_interval == 0:
            torch.save(decoder.state_dict(), save_dir / f"decoder_epoch_{epoch + 1}.pth")
            torch.save(optimizer.state_dict(), save_dir / f"optimizer_epoch_{epoch + 1}.pth")

            with torch.no_grad():
                output = torch.concat([content_batch, style_batch, g], dim=0)
                save_image(output, save_dir / f"output_epoch_{epoch + 1}.png", nrow=args.batch_size, normalize=True)

if __name__ == "__main__":
    main()