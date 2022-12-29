from pathlib import Path
from PIL import Image
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file

script_dir = Path(__file__).parent.resolve()

model_file = script_dir/'models/astropi-day-vs-night.tflite' # name of model
data_dir = script_dir/'data'
label_file = data_dir/'day-vs-night.txt' # Name of your label file
image_file = data_dir/'tests'/'day_3.jpg' # Name of image for classification

# Load the labels
labels = read_label_file(label_file)

# Load the model
interpreter = make_interpreter(model_file)
interpreter.allocate_tensors()

# Load the image
image = Image.open(image_file).convert('RGB').resize((224, 224), Image.ANTIALIAS)

# Run the inference
_, scale = common.set_resized_input(interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
interpreter.invoke()
classes = classify.get_classes(interpreter, top_k=1, score_threshold=0.0)

# Print the results
for c in classes:
    print('%s: %.5f' % (labels.get(c.id, c.id), c.score))

# Output:
# day: 0.99999

