import wandb
import matplotlib.pyplot as plt

def wandb_imshow(img, caption):
    fig = plt.figure()
    plt.imshow(img.detach().cpu(), 'gray')
    plt.colorbar()
    wandb.log({caption: fig})
    plt.close()