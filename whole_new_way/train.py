

def main(args):
    return

if __name__== 'main':
    parser = ArgumentParser()
    
    parser.add_argument('--data-dir', required=True, help='Mapillary directory')
    parser.add_argument('--model', required=True, choices=['resnet_oc_lw', 'resnet_oc'], help='Tell me what to train')
    parser.add_argument('--height', type=int, default=1080, help='Height of images, nothing to add')
    parser.add_argument('--num-epochs', type=int, default=10, help='If you use resume, give a number considering for how long it trained')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--save-dir', required=True, help='Where to save your model')
    parser.add_argument('--pretrained', action='store_true', help='Whether to use pretrained backbone')
    parser.add_argument('--resume', action='store_true', help='Resumes from the last save from --savedir directory')
    parser.add_argument('--pretrained', action='store_true', help='Whether to log metrics to wandb')    
    parser.add_argument('--project-name', default='Junk', help='Project name for weights and Biases')
    parser.add_argument('--gpu-ids', type=str, default='0', help='use which gpu to train, must be a \
                                                                comma-separated list of integers only (default=0)')
    parser.add_argument('--epochs-save', type=int, default=3, help='You can use this value to save model every X epochs')
    main(parser.parse_args())