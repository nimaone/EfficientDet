import numpy as np
import random
import warnings
import cv2
from tensorflow import keras
import albumentations as A

from utils.anchors import anchors_for_shape, anchor_targets_bbox, AnchorParameters


class Generator(keras.utils.Sequence):
    """
    Abstract generator class.
    """

    def __init__(
            self,
            phi=0,
            image_sizes=(512, 640, 768, 896, 1024, 1280, 1408),
            misc_effect=None,
            visual_effect=None,
            use_augmentations=None,
            horizontal_flip = None, 
            vertical_flip = None, 
            RandomBrightnessContrast = None, 
            RandomColorShift = None,
            RandomRotate90 = None,
            batch_size=1,
            group_method='random',  # one of 'none', 'random', 'ratio'
            shuffle_groups=True,
            detect_text=False,
            detect_quadrangle=False,
    ):
        """
        Initialize Generator object.

        Args:
            batch_size: The size of the batches to generate.
            group_method: Determines how images are grouped together (defaults to 'ratio', one of ('none', 'random', 'ratio')).
            shuffle_groups: If True, shuffles the groups each epoch.
            image_sizes:
        """
        self.misc_effect = misc_effect
        self.visual_effect = visual_effect
        self.use_augmentations = use_augmentations
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.RandomBrightnessContrast = RandomBrightnessContrast,
        self.RandomColorShift = RandomColorShift
        self.RandomRotate90 = RandomRotate90
        self.batch_size = int(batch_size)
        self.group_method = group_method
        self.shuffle_groups = shuffle_groups
        self.detect_text = detect_text
        self.detect_quadrangle = detect_quadrangle
        self.image_size = image_sizes[phi]
        self.groups = None
        self.anchor_parameters = AnchorParameters.default if not self.detect_text else AnchorParameters(
            ratios=(0.25, 0.5, 1., 2.),
            sizes=(16, 32, 64, 128, 256))
        self.anchors = anchors_for_shape((self.image_size, self.image_size), anchor_params=self.anchor_parameters)
        self.num_anchors = self.anchor_parameters.num_anchors()

        # Define groups
        self.group_images()

        # Shuffle when initializing
        if self.shuffle_groups:
            random.shuffle(self.groups)

    def on_epoch_end(self):
        if self.shuffle_groups:
            random.shuffle(self.groups)

    def size(self):
        """
        Size of the dataset.
        """
        raise NotImplementedError('size method not implemented')

    def get_anchors(self):
        """
        loads the anchors from a txt file
        """
        with open(self.anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        # (N, 2), wh
        return np.array(anchors).reshape(-1, 2)

    def num_classes(self):
        """
        Number of classes in the dataset.
        """
        raise NotImplementedError('num_classes method not implemented')

    def has_label(self, label):
        """
        Returns True if label is a known label.
        """
        raise NotImplementedError('has_label method not implemented')

    def has_name(self, name):
        """
        Returns True if name is a known class.
        """
        raise NotImplementedError('has_name method not implemented')

    def name_to_label(self, name):
        """
        Map name to label.
        """
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        """
        Map label to name.
        """
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        """
        Compute the aspect ratio for an image with image_index.
        """
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):
        """
        Load annotations for an image_index.
        """
        raise NotImplementedError('load_annotations method not implemented')

    def load_annotations_group(self, group):
        """
        Load annotations for all images in group.
        """
        annotations_group = [self.load_annotations(image_index) for image_index in group]
        for annotations in annotations_group:
            assert (isinstance(annotations,
                               dict)), '\'load_annotations\' should return a list of dictionaries, received: {}'.format(
                type(annotations))
            assert (
                    'labels' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'
            assert (
                    'bboxes' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'

        return annotations_group

    def filter_annotations(self, image_group, annotations_group, group):
        """
        Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                (annotations['bboxes'][:, 0] < 0) |
                (annotations['bboxes'][:, 1] < 0) |
                (annotations['bboxes'][:, 2] <= 0) |
                (annotations['bboxes'][:, 3] <= 0) |
                (annotations['bboxes'][:, 2] > image.shape[1]) |
                (annotations['bboxes'][:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    annotations['bboxes'][invalid_indices, :]
                ))
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], invalid_indices, axis=0)
            # if annotations['bboxes'].shape[0] == 0:
            #     warnings.warn('Image with id {} (shape {}) contains no valid boxes before transform'.format(
            #         group[index],
            #         image.shape,
            #     ))
        return image_group, annotations_group

    def clip_transformed_annotations(self, image_group, annotations_group):
        """
        Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        filtered_image_group = []
        filtered_annotations_group = []
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            image_height = image.shape[0]
            image_width = image.shape[1]
            # x1
#             annotations['bboxes'][:, 0] = np.clip(annotations['bboxes'][:, 0], 0, image_width - 2)
#             # y1
#             annotations['bboxes'][:, 1] = np.clip(annotations['bboxes'][:, 1], 0, image_height - 2)
#             # x2
#             annotations['bboxes'][:, 2] = np.clip(annotations['bboxes'][:, 2], 1, image_width - 1)
#             # y2
#             annotations['bboxes'][:, 3] = np.clip(annotations['bboxes'][:, 3], 1, image_height - 1)
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            out_indices = np.where(
                ((annotations['bboxes'][:, 2] + annotations['bboxes'][:, 0])/2 < 0) |
                ((annotations['bboxes'][:, 3] + annotations['bboxes'][:, 1])/2 < 0) |
                ((annotations['bboxes'][:, 2] + annotations['bboxes'][:, 0])/2 > image_width) |
                ((annotations['bboxes'][:, 3] + annotations['bboxes'][:, 1])/2 > image_height)
            )[0]
            small_indices = np.where(
                (annotations['bboxes'][:, 2] - annotations['bboxes'][:, 0] < 3) |
                (annotations['bboxes'][:, 3] - annotations['bboxes'][:, 1] < 3)
            )[0]

            # delete invalid indices
            if len(small_indices) or len(out_indices):
#                 print('!!!!!!!!!!')
                
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], small_indices, axis=0)
                    annotations_group[index][k] = np.delete(annotations[k], out_indices, axis=0)
                    # print(annotations['bboxes'][out_indices])
                # import cv2
                # for invalid_index in small_indices:
                #     x1, y1, x2, y2 = annotations['bboxes'][invalid_index]
                #     label = annotations['labels'][invalid_index]
                #     class_name = self.labels[label]
                #     print('width: {}'.format(x2 - x1))
                #     print('height: {}'.format(y2 - y1))
                #     cv2.rectangle(image, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (0, 255, 0), 2)
                #     cv2.putText(image, class_name, (int(round(x1)), int(round(y1))), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
                # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                # cv2.imshow('image', image)
                # cv2.waitKey(0)
            filtered_image_group.append(image)
            filtered_annotations_group.append(annotations_group[index])

        return filtered_image_group, filtered_annotations_group

    def load_image_group(self, group):
        """
        Load images for all images in a group.
        """
        return [self.load_image(image_index) for image_index in group]

    def augmentations_pipeline(self,image_group, annotations_group):

        # augmentations_list = list()    

        # if self.horizontal_flip:
        #     H = A.HorizontalFlip(p=0.5)
        #     augmentations_list.append(H)

        # if self.vertical_flip:
        #     V = A.VerticalFlip(p=0.5)
        #     augmentations_list.append(V)

        # if self.RandomBrightnessContrast:
        #     B =  A.OneOf([
        #         A.RandomBrightnessContrast(brightness_limit=0.3,
        #                                    contrast_limit=0.2,
        #                                    p=.8),
        #         A.RandomGamma(gamma_limit=(80, 120))
        #         ],p=.8)
        #     augmentations_list.append(B) 

        # if self.RandomColorShift:
        #     C = A.OneOf([
        #         A.HueSaturationValue(hue_shift_limit=.2, sat_shift_limit=.2,
        #                              val_shift_limit=0.2,p=.8), 
        #         A.RGBShift(r_shift_limit=20, b_shift_limit=15, g_shift_limit=15)
        #         ])
        #     augmentations_list.append(C)
        # augmentations_list.append(A.CLAHE(p=0.8))
        # augmentations_list.append(A.ToGray(p=0.01))
        
        # if self.RandomRotate90:
        #     R = A.RandomRotate90(p=0.5)
        #     augmentations_list.append(R)   
        augmentations_list_1= [
#             A.Resize(height=1024, width=1024, p=1),
#             A.RandomSizedCrop(min_max_height=(824, 824), height=1024, width=1024, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2, p=0.9),
                  ],p=0.9),

#         augmentations_list_2 = [         
            A.Rotate (limit=list(np.arange(-90, 90+16, 15)), interpolation=1, border_mode=0, p=0.8),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=640, width=640, p=1),
        ]
        
            
        transform_1 = A.Compose(augmentations_list_1, 
                                keypoint_params=A.KeypointParams(
                                    format='xy',remove_invisible=False),
                                p=1
                            )
#         transform_2 = A.Compose(augmentations_list_2, 
#                                 keypoint_params=A.KeypointParams(
#                                     format='xy',remove_invisible=False),
#                                 p=1
#                             )                        
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # preprocess a single group entry
            quadrangles = annotations['quadrangles'].reshape(-1,2)


            transformed = transform_1(image=image,
                            keypoints=(quadrangles)
                                    )
            aug_quadrangles = np.array(transformed['keypoints']).reshape(-1,4,2).astype(np.float32)
            xmin = np.min(aug_quadrangles, axis=1)[:, 0]
            ymin = np.min(aug_quadrangles, axis=1)[:, 1]
            xmax = np.max(aug_quadrangles, axis=1)[:, 0]
            ymax = np.max(aug_quadrangles, axis=1)[:, 1]
            annotations['bboxes'] = np.stack([xmin,ymin,xmax,ymax],axis=1)
            annotations['quadrangles'] = aug_quadrangles
            image_group[index] = transformed['image']
#         image_group, annotations_group = self.clip_transformed_annotations(image_group, annotations_group)

#         for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
#             # preprocess a single group entry
#             quadrangles = annotations['quadrangles'].reshape(-1,2)


#             transformed = transform_2(image=image,
#                             keypoints=(quadrangles)
#                                     )
#             aug_quadrangles = np.array(transformed['keypoints']).reshape(-1,4,2).astype(np.float32)
#             xmin = np.min(aug_quadrangles, axis=1)[:, 0]
#             ymin = np.min(aug_quadrangles, axis=1)[:, 1]
#             xmax = np.max(aug_quadrangles, axis=1)[:, 0]
#             ymax = np.max(aug_quadrangles, axis=1)[:, 1]
#             annotations['bboxes'] = np.stack([xmin,ymin,xmax,ymax],axis=1)
#             annotations['quadrangles'] = aug_quadrangles
#             image_group[index] = transformed['image']
        return image_group, annotations_group
    

    def random_visual_effect_group_entry(self, image, annotations):
        """
        Randomly transforms image and annotation.
        """
        # apply visual effect
        image = self.visual_effect(image)
        return image, annotations

    def random_visual_effect_group(self, image_group, annotations_group):
        """
        Randomly apply visual effect on each image.
        """
        assert (len(image_group) == len(annotations_group))

        if self.visual_effect is None:
            # do nothing
            return image_group, annotations_group

        for index in range(len(image_group)):
            # apply effect on a single group entry
            image_group[index], annotations_group[index] = self.random_visual_effect_group_entry(
                image_group[index], annotations_group[index]
            )

        return image_group, annotations_group

    def random_misc_group_entry(self, image, annotations):
        """
        Randomly transforms image and annotation.
        """
        # randomly transform both image and annotations
        image, annotations = self.misc_effect(image, annotations)
        return image, annotations

    def random_misc_group(self, image_group, annotations_group):
        """
        Randomly transforms each image and its annotations.
        """

        assert (len(image_group) == len(annotations_group))

        if self.misc_effect is None:
            return image_group, annotations_group

        for index in range(len(image_group)):
            # transform a single group entry
            image_group[index], annotations_group[index] = self.random_misc_group_entry(image_group[index],
                                                                                        annotations_group[index])

        return image_group, annotations_group

    def preprocess_group_entry(self, image, annotations):
        """
        Preprocess image and its annotations.
        """

        # preprocess the image
        image, scale = self.preprocess_image(image)
        # print(image.shape)
        # print(scale)

        annotations['bboxes'] *= scale
        if self.detect_quadrangle:
            annotations['quadrangles'] *= scale
            quadrangles = np.array([self.xywhtheta_to_coords(
                                    self.coords_to_xywhtheta(q.reshape(1,8)))
                                    for q in annotations['quadrangles']
                                                            ]
                                      ).reshape(-1,4,2)
            quadrangles = np.array([self.reorder_vertexes(q) for q in quadrangles])
            annotations['quadrangles'] = quadrangles 
            # print(quadrangles.shape)
        return image,annotations
        # apply resizing to annotations too
    
    def preprocess_group(self, image_group, annotations_group):
        """
        Preprocess each image and its annotations in its group.
        """
        assert (len(image_group) == len(annotations_group))
        # print(len(image_group))
        # print(len(annotations_group))

        for index,(image,annotations) in enumerate( zip(image_group,annotations_group)):
            # preprocess a single group entry
            image_group[index], annotations_group[index] = self.preprocess_group_entry(image,annotations)

        return image_group, annotations_group
    def xywhtheta_to_coords(self,coordinate, with_label=False):
        """
        :param coordinate: format [x_c, y_c, w, h, theta]
        :return: format [x1, y1, x2, y2, x3, y3, x4, y4]
        """

        boxes = []
        if with_label:
            for rect in coordinate:
                box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
                box = np.reshape(box, [-1, ])
                boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7], rect[5]])
        else:
            for rect in coordinate:
                box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
                box = np.reshape(box, [-1, ])
                boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]])

        return np.array(boxes, dtype=np.float32)


    def coords_to_xywhtheta(self,coordinate, with_label=False):
        """
        :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4, (label)]
        :param with_label: default True
        :return: format [x_c, y_c, w, h, theta, (label)]
        """

        boxes = []
        if with_label:
            for rect in coordinate:
                box = np.int0(rect[:-1])
                box = box.reshape([4, 2])
                rect1 = cv2.minAreaRect(box)

                x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]

                if theta == 0:
                    w, h = h, w
                    theta -= 90

                boxes.append([x, y, w, h, theta, rect[-1]])

        else:
            for rect in coordinate:
                box = np.int0(rect)
                box = box.reshape([4, 2])
                rect1 = cv2.minAreaRect(box)

                x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]

                if theta == 0:
                    w, h = h, w
                    theta -= 90

                boxes.append([x, y, w, h, theta])

        return np.array(boxes, dtype=np.float32)
    def reorder_vertexes(self, vertexes):
        """
        reorder vertexes as the paper shows, (top, right, bottom, left)
        Args:
            vertexes:

        Returns:

        """
        assert vertexes.shape == (4, 2)
        xmin, ymin = np.min(vertexes, axis=0)
        xmax, ymax = np.max(vertexes, axis=0)

        # determine the first point with the smallest y,
        # if two vertexes has same y, choose that with smaller x,
        ordered_idxes = np.argsort(vertexes, axis=0)
        ymin1_idx = ordered_idxes[0, 1]
        ymin2_idx = ordered_idxes[1, 1]
        if vertexes[ymin1_idx, 1] == vertexes[ymin2_idx, 1]:
            if vertexes[ymin1_idx, 0] <= vertexes[ymin2_idx, 0]:
                first_vertex_idx = ymin1_idx
            else:
                first_vertex_idx = ymin2_idx
        else:
            first_vertex_idx = ymin1_idx
        ordered_idxes = [(first_vertex_idx + i) % 4 for i in range(4)]
        ordered_vertexes = vertexes[ordered_idxes]
        # drag the point to the corresponding edge
        ordered_vertexes[0, 1] = ymin
        ordered_vertexes[1, 0] = xmax
        ordered_vertexes[2, 1] = ymax
        ordered_vertexes[3, 0] = xmin
        return ordered_vertexes
    
    def group_images(self):
        """
        Order the images according to self.order and makes groups of self.batch_size.
        """
        # determine the order of the images

        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                       range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group, annotations_group):
        """
        Compute inputs for the network using an image_group.
        """
        batch_images = np.array(image_group).astype(np.float32)
        return [batch_images]

    def compute_alphas_and_ratios(self, annotations_group):
        for i, annotations in enumerate(annotations_group):
            
            # print(quadrangles)
            quadrangles = annotations['quadrangles']
            alphas = np.zeros((quadrangles.shape[0], 2), dtype=np.float32)
            xmin = np.min(quadrangles, axis=1)[:, 0]
            ymin = np.min(quadrangles, axis=1)[:, 1]
            xmax = np.max(quadrangles, axis=1)[:, 0]
            ymax = np.max(quadrangles, axis=1)[:, 1]
            annotations['bboxes'] = np.vstack([xmin,ymin,xmax,ymax]).T
            # alpha1, alpha2, alpha3, alpha4
            alphas[:, 0] = (quadrangles[:, 0, 0] - xmin) / (xmax - xmin)
            alphas[:, 1] = (quadrangles[:, 1, 1] - ymin) / (ymax - ymin)
#             alphas[:, 2] = (xmax - quadrangles[:, 2, 0]) / (xmax - xmin)
#             alphas[:, 3] = (ymax - quadrangles[:, 3, 1]) / (ymax - ymin)
            annotations['alphas'] = alphas
            # ratio
            area1 = 0.5 * alphas[:, 0] * (1 - alphas[:, 1])
            area2 = 0.5 * alphas[:, 1] * (1 - alphas[:, 0])
#             area3 = 0.5 * alphas[:, 2] * (1 - alphas[:, 1])
#             area4 = 0.5 * alphas[:, 3] * (1 - alphas[:, 2])
            annotations['ratios'] = 1 - area1 - area2# - area3 - area4
        return annotations_group
    def compute_targets(self, image_group, annotations_group):
        """
        Compute target outputs for the network using images and their annotations.
        """
        """
        Compute target outputs for the network using images and their annotations.
        """
#         print(annotations_group)
        batches_targets = anchor_targets_bbox(
            self.anchors,
            image_group,
            annotations_group,
            num_classes=self.num_classes(),
            detect_quadrangle=self.detect_quadrangle
        )
        return list(batches_targets)

    def compute_inputs_targets(self, group, debug=False):
        """
        Compute inputs and target outputs for the network.
        """

        # load images and annotations
        # list
        image_group = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        # image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # randomly apply visual effect
        image_group, annotations_group = self.random_visual_effect_group(image_group, annotations_group)

        # randomly transform data
        # image_group, annotations_group = self.random_transform_group(image_group, annotations_group)

        # randomly apply misc effect
        image_group, annotations_group = self.random_misc_group(image_group, annotations_group)
        
        if self.use_augmentations:

            image_group, annotations_group = self.augmentations_pipeline(image_group, annotations_group)

        # perform preprocessing steps
        # print((image_group)[0])
        # print((annotations_group)[0])

        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # check validity of annotations
        image_group, annotations_group = self.clip_transformed_annotations(image_group, annotations_group)

        assert len(image_group) != 0
        assert len(image_group) == len(annotations_group)

        if self.detect_quadrangle:
            # compute alphas and ratio for targets
            annotations_group = self.compute_alphas_and_ratios(annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group, annotations_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        if debug:
            return inputs, targets, annotations_group

        return inputs, targets

    def __len__(self):
        """
        Number of batches for generator.
        """

        return len(self.groups)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        group = self.groups[index]
        inputs, targets = self.compute_inputs_targets(group)
        return inputs, targets

    def preprocess_image(self, image):
        # image, RGB
        image_height, image_width = image.shape[:2]
        if image_height > image_width:
            scale = self.image_size / image_height
            resized_height = self.image_size
            resized_width = int(image_width * scale)
        else:
            scale = self.image_size / image_width
            resized_height = int(image_height * scale)
            resized_width = self.image_size

        image = cv2.resize(image, (resized_width, resized_height))
        image = image.astype(np.float32)
        image /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image -= mean
        image /= std
        pad_h = self.image_size - resized_height
        pad_w = self.image_size - resized_width
        image = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant')
        return image, scale

    def get_augmented_data(self, group):
        """
        Compute inputs and target outputs for the network.
        """

        # load images and annotations
        # list
        image_group = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # randomly apply visual effect
        # image_group, annotations_group = self.random_visual_effect_group(image_group, annotations_group)

        # randomly transform data
        # image_group, annotations_group = self.random_transform_group(image_group, annotations_group)

        # randomly apply misc effect
        # image_group, annotations_group = self.random_misc_group(image_group, annotations_group)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # check validity of annotations
        image_group, annotations_group = self.clip_transformed_annotations(image_group, annotations_group, group)

        assert len(image_group) != 0
        assert len(image_group) == len(annotations_group)

        # compute alphas for targets
        self.compute_alphas_and_ratios(annotations_group)

        return image_group, annotations_group
