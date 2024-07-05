import sys
import argparse
from jetson_inference import imageNet
from jetson_utils import cudaFont, cudaAllocMapped

# parse the command line
parser = argparse.ArgumentParser(description="Run an ONNX model using imageNet.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=imageNet.Usage())

parser.add_argument("--model", type=str, default="model.onnx", help="path to the ONNX model")
parser.add_argument("--input_blob", type=str, default="input_0", help="name of the input blob")
parser.add_argument("--output_blob", type=str, default="output_0", help="name of the output blob")
parser.add_argument("--width", type=int, default=224, help="input width")
parser.add_argument("--height", type=int, default=224, help="input height")
parser.add_argument("--topK", type=int, default=1, help="show the topK number of class predictions (default: 1)")

try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# load the recognition network
net = imageNet(model=args.model, input_blob=args.input_blob, output_blob=args.output_blob)

# allocate memory for a dummy input image
img = cudaAllocMapped(width=args.width, height=args.height, format='rgb8')

# run dummy inference
class_id, confidence = net.Classify(img)

# get the class description
class_desc = net.GetClassDesc(class_id)

# create a font for overlaying text
font = cudaFont()

# overlay the result on the image
font.OverlayText(img, text=f"{confidence*100:.2f}% {class_desc}", x=5, y=5, color=font.White, background=font.Gray40)

print(f"Classified dummy image with confidence {confidence*100:.2f}% ({class_desc})")
