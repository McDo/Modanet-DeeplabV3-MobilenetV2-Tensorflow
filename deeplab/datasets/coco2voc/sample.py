from coco2voc import *
from PIL import Image
import argparse


def on_press(event):
	"""
	Keyboard interaction ,key `a` for next image, key `d` for previous image, key `t` segmentation toggle
	:param event: :class:`~matplotlib.backend_bases.KeyEvent`, keyboard event
	:return: None
	"""
	global i, l, frames, segs, fplot, splot, fig, ax, s_toggle, id_list, figsizes

	if event.key == 'd':
		i = (i+1) % l
		s_toggle = True
		splot.set_alpha(0.4)
	elif event.key == 'a':
		i = (i-1) % l
		s_toggle = True
		splot.set_alpha(0.4)
	elif event.key == 't':
		# show or hide segmentation
		s_toggle = not s_toggle
		splot.set_alpha(0.4) if s_toggle else splot.set_alpha(0)

	fplot.set_data(frames[i])
	splot.set_data(segs[i])

	fig.set_size_inches(figsizes[i], forward=True)
	fig.canvas.draw()
	ax.set_title(id_list[i])

	pass


def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser(description='Convert coco format data to voc format for deeplab training')
	parser.add_argument('--dataset', dest='dataset',
						help='directory training dataset',
						type=str, required=True)
	parser.add_argument('--mode', dest='mode',
						help='train or val',
						type=str, choices=['train', 'val'], required=True)
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_args()
	mode = args.mode
	root = args.dataset
	target_folder = os.path.join(root, 'voc')
	instance_target_path = os.path.join(target_folder, 'SegmentationObject')
	class_target_path = os.path.join(target_folder, 'SegmentationClass')
	id_target_path = os.path.join(target_folder, 'SegmentationId')
	data_folder = os.path.join(root, 'images', mode)
	annotations_file = os.path.join(root, 'annotations', f'instances_{mode}.json')
	image_id_list_path = os.path.join(target_folder, 'ImageSets', 'Segmentation', f'{mode}.txt')

	# Load an image with it's id segmentation and show
	coco = COCO(annotations_file)

	# Read ids of images whose annotations have been converted from specified file
	with open(image_id_list_path) as f:
		id_list = [line.split()[0] for line in f]

	i = 0
	l = len(id_list)
	frames = []
	segs =[]
	figsizes = []
	s_toggle = True
	dpi = 100

	for id in id_list:
		# Get the image's file name and load image from data folder
		img_ann = coco.loadImgs(int(id))

		file_name = img_ann[0]['file_name']
		# im_data = plt.imread(os.path.join(data_folder, file_name))
		im_data = np.asarray(Image.open(os.path.join(data_folder, file_name)).convert('RGB'))
		height, width, depth = im_data.shape
		frames.append(im_data)

		size = width / float(dpi), height / float(dpi)
		figsizes.append(size)

		# Load segmentation - note that the loaded '.npz' file is a dictionary, and the data is at key 'arr_0'
		id_seg = np.load(os.path.join(id_target_path, id +'.npz'))
		segs.append(id_seg['arr_0'])
		
		# Example for loading class or instance segmentations
		instance_filename = os.path.join(instance_target_path, id +'.png')
		class_filename = os.path.join(class_target_path, id +'.png')
		instance_seg = np.array(Image.open(instance_filename))
		class_seg = np.array(Image.open(class_filename))

	# Show image with segmentations
	fig, ax = plt.subplots(figsize=figsizes[0], dpi=dpi)
	fig.canvas.mpl_connect('key_press_event', on_press)

	fplot = ax.imshow(frames[i%l])
	splot = ax.imshow(segs[i%l], alpha=0.4)

	ax.set_aspect(aspect='auto')# must after imshow

	plt.tight_layout()
	plt.axis('off')
	plt.show()
