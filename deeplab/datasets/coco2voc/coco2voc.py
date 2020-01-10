from pycocotools.coco import COCO
from coco2voc_aux import *
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import time
import labelme


def coco2voc(anns_file, root_folder, mode='train', name_bits=7, n=None, compress=True):
	'''
	This function converts COCO style annotations to PASCAL VOC style instance and class
		segmentations. Additionaly, it creates a segmentation mask(1d ndarray) with every pixel contatining the id of
		the instance that the pixel belongs to.
	:param anns_file: COCO annotations file, as given in the COCO data set
	:param root_folder: path to the folder where the results will be saved
	:param mode: `train`, 'trainval' or `val`
	:param n: Number of image annotations to convert. Default is None in which case all of the annotations are converted
	:param compress: if True, id segmentation masks are saved as '.npz' compressed files. if False they are saved as '.npy'
	:return: All segmentations are saved to the target folder, along with a list of ids of the images that were converted
	'''

	coco_instance = COCO(anns_file)
	coco_imgs = coco_instance.imgs

	if n is None:
		n = len(coco_imgs)
	else:
		assert type(n) == int, "n must be an int"
		n = min(n, len(coco_imgs))

	instance_target_path = os.path.join(root_folder, 'SegmentationObject')
	class_target_path = os.path.join(root_folder, 'SegmentationClass')
	id_target_path = os.path.join(root_folder, 'SegmentationId')
	image_id_list_path = os.path.join(root_folder, 'ImageSets', 'Segmentation', f'{mode}.txt')

	if not os.path.exists(instance_target_path): os.system(f'mkdir {instance_target_path}')
	if not os.path.exists(class_target_path): os.system(f'mkdir {class_target_path}')
	if not os.path.exists(id_target_path): os.system(f'mkdir {id_target_path}')
	image_id_list = open(image_id_list_path, 'a+')

	start = time.time()
	for i, img in enumerate(coco_imgs):

		anns_ids = coco_instance.getAnnIds(img)
		anns = coco_instance.loadAnns(anns_ids)
		if not anns:
			continue

		img = str(img).zfill(name_bits)
		class_seg, instance_seg, id_seg = annsToSeg(anns, coco_instance)

		# Image.fromarray(class_seg).convert("L").save(class_target_path + '/' + str(img) + '.png')
		# Image.fromarray(instance_seg).convert("L").save(instance_target_path + '/' + str(img) + '.png')
		labelme.utils.lblsave(class_target_path + '/' + str(img) + '.png', class_seg)
		labelme.utils.lblsave(instance_target_path + '/' + str(img) + '.png', instance_seg)
		
		if compress:
			np.savez_compressed(os.path.join(id_target_path, str(img)), id_seg)
		else:
			np.save(os.path.join(id_target_path, str(img) + '.npy'), id_seg)

		image_id_list.write(str(img) + '\n')

		if i % 100 == 0 and i > 0:
			print(str(i) + " annotations processed" +
				  " in " + str(int(time.time()-start)) + " seconds")
		if i >= n:
			break

	image_id_list.close()
	return