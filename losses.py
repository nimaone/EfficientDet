"""
Copyright 2017-2018 Fizyr (https://fizyr.com)
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# import keras
import numpy as np
import math
import polyiou
import cv2
from tensorflow import keras
import tensorflow as tf
from utils.anchors import anchors_for_shape,AnchorParameters
phi=4
image_sizes=(512, 640, 768, 896, 640, 1280, 1408)
image_size = (image_sizes[phi],image_sizes[phi])

import math as m
pi = tf.constant(m.pi)


def change_angle_tf(c):
    # c=cv2.minAreaRect(c)
    # angle=c[2]
    # w=c[1][0]
    # h=c[1][1]
    x1=c[0][0]
    x2=c[1][0]
    x3=c[2][0]
    x4=c[3][0]
    y1=c[0][1]
    y2=c[1][1]
    y3=c[2][1]
    y4=c[3][1]
    h = tf.sqrt(tf.pow(x2-x1, 2) + tf.pow(y2-y1, 2))
    w = tf.sqrt(tf.pow(x3-x2, 2) + tf.pow(y3-y2, 2))
    angle = tf.atan2(y4 - y1, x4 - x1)*180./pi
    if angle<-90:
        angle=angle+90
    
    if angle>+90:
        angle=angle-180
    # print('ghghgh',angle)
    if w<h:
      if angle < -45:
          angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
      else:
        angle = -angle  
    else:
      if angle < -45:
          angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
      else:
        angle = -angle
    return tf.constant(angle)

def rotate_tf(ct,angle):
    rotation_amount_rad = angle * pi / 180.0
    c1=tf.reduce_mean(ct,axis=0)
    a=tf.cos(rotation_amount_rad)
    b=tf.sin(rotation_amount_rad)
    M=tf.reshape(tf.stack((a , b,  (1-a)*c1[0] -  b*c1[1],
                        -b , a,   b*c1[0]    + (1-a)*c1[1])
            ),(2,3))
    ones = tf.ones(shape=(tf.shape(ct)[0], 1))
    points_ones = tf.concat((ct, ones),axis=1)
    
    # transform points
    return  tf.transpose(tf.matmul(M,tf.transpose(points_ones)))
              



def quad_overlaps_ciou(decoded_boxes_quad,regression_target_quad):
        def quad_overlaps_ciou_(rt_quad,decode_quad):
          return  np.array([polyiou.iou_poly(decode_quad[i].astype(np.float64), r) 
                    for i,r in enumerate( rt_quad.astype(np.float64)) ])


        return tf.numpy_function(
                quad_overlaps_ciou_, [regression_target_quad,decoded_boxes_quad], Tout=tf.float64,
                )

def quad_to_off_bbox(ct):    
    x1=ct[:,0]
    x2=ct[:,2]
    x3=ct[:,4]
    x4=ct[:,6]
    y1=ct[:,1]
    y2=ct[:,3]
    y3=ct[:,5]
    y4=ct[:,7]
    # angle = tf.cast(tf.atan2(y4 - y1, x4 - x1)*180./np.pi,tf.float64)
    angle = tf.atan2(y4 - y1, x4 - x1)*180./np.pi

    ctx=tf.reduce_mean((x1,x2,x3,x4),axis=0)
    cty=tf.reduce_mean((y1,y2,y3,y4),axis=0)


    angle=tf.where(tf.less(angle,-90),angle+90,angle)
    angle=tf.where(tf.greater(angle,+90),angle-180,angle)
    angles=tf.where(tf.less(angle,-45),-(90 + angle),-angle)
    angles=-angles
    rotation_amount_rad = angles * np.pi / 180.0
    # a=(tf.cast(tf.cos(rotation_amount_rad),tf.float32))
    # b=(tf.cast(tf.sin(rotation_amount_rad),tf.float32))

    a=tf.cos(rotation_amount_rad)
    b=tf.sin(rotation_amount_rad)

    offset = tf.stack(((1-a)*ctx   -  b*cty,
              b*ctx      + (1-a)*cty   ),axis=1)
    M=tf.stack((
              tf.stack((a , b),axis=1),
              tf.stack((-b , a,),axis=1),
              
              ),axis=1)


    xs=tf.stack((x1,x2,x3,x4),axis=1)
    ys=tf.stack((y1,y2,y3,y4),axis=1)
    poins = tf.stack((xs,ys),axis=1)
    rots = tf.matmul(M,poins)

    result=tf.transpose(rots,[0,2,1])+ tf.expand_dims( offset,axis=1)
    xmin= tf.reduce_min(result[...,0],axis=1)
    ymin= tf.reduce_min(result[...,1],axis=1)
    xmax= tf.reduce_max(result[...,0],axis=1)
    ymax= tf.reduce_max(result[...,1],axis=1)
    return tf.stack((xmin,ymin,xmax,ymax),axis=1)



def bbox_overlaps_ciou(bboxes1, bboxes2):

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = tf.zeros(1)
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = tf.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2
  
    inter_max_xy = tf.reduce_min((bboxes1[:, 2:],bboxes2[:, 2:]),axis=0)
    inter_min_xy = tf.reduce_max((bboxes1[:, :2],bboxes2[:, :2]),axis=0)
    out_max_xy = tf.reduce_max((bboxes1[:, 2:],bboxes2[:, 2:]),axis=0)
    out_min_xy = tf.reduce_min((bboxes1[:, :2],bboxes2[:, :2]),axis=0)
    
    # inter = tf.clip_by_value((inter_max_xy - inter_min_xy), 0,np.inf)
    inter = (inter_max_xy - inter_min_xy)
    

    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    # inter_diag = tf.sqrt((center_x2 - center_x1)**2 + (center_y2 - center_y1)**2)/image_size[0]

    # outer = tf.clip_by_value((out_max_xy - out_min_xy), 0,np.inf)
    outer = (out_max_xy - out_min_xy)


    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    # outer_diag = tf.sqrt((outer[:, 0] ** 2) + (outer[:, 1] ** 2))/image_size[0]
    union = area1+area2-inter_area
    u = (inter_diag) / outer_diag
    # u = 2*(inter_diag * outer_diag)

    iou = inter_area / union+1e-7
    v = (4 / (math.pi ** 2)) * tf.math.pow((tf.math.atan(w2 / h2) - tf.math.atan(w1 / h1)), 2)
    S = 1 - iou
    S=tf.stop_gradient(S)
    alpha = v / (S + v)
    alpha=tf.stop_gradient(alpha)
    cious = iou - (u + alpha * v)
    cious = tf.clip_by_value(cious,-1.0, 1.0)
    if exchange:
        cious = cious.T
    # with tf.Session() as ss:
      # ss.run(init)
      # print('w1',ss.run(w1))
      # print('area1',ss.run(area1))
      # print('bboxes2[:, 2:]',ss.run(bboxes2[:, 2:]))
      # print('inter_max_xy\n',ss.run(inter_max_xy))
      # print('cious\n',np.sum(ss.run(cious)))
    return cious
def bbox_transform_inv(boxes, deltas, scale_factors=None):
    cxa = (boxes[..., 0] + boxes[..., 2]) / 2
    cya = (boxes[..., 1] + boxes[..., 3]) / 2
    wa = boxes[..., 2] - boxes[..., 0]
    ha = boxes[..., 3] - boxes[..., 1]
    ty, tx, th, tw = deltas[..., 0], deltas[..., 1], deltas[..., 2], deltas[..., 3]
    # print(deltas.shape)
    talpha = deltas[:,4:6]
    # print(talpha.shape)
    if scale_factors:
        ty *= scale_factors[0]
        tx *= scale_factors[1]
        th *= scale_factors[2]
        tw *= scale_factors[3]
    w = tf.exp(tw) * wa
    h = tf.exp(th) * ha
    cy = ty * ha + cya
    cx = tx * wa + cxa
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    boxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
    
    talphas = 1 / (1 + tf.exp(-talpha))

    a0 = (talphas[ ..., 0])
    a1 = (talphas[ ..., 1])
   
    
    quadrangles0 = boxes[ ..., 0] + (boxes[ ..., 2] - boxes[ ..., 0]) * a0
    quadrangles1 = boxes[ ..., 1]
    quadrangles2 = boxes[ ..., 2]
    quadrangles3 = boxes[ ..., 1] + (boxes[ ..., 3] - boxes[ ..., 1]) * a1
    quadrangles4 = boxes[ ..., 2] - (boxes[ ..., 2] - boxes[ ..., 0]) * a0
    quadrangles5 = boxes[ ..., 3]
    quadrangles6 = boxes[ ..., 0]
    quadrangles7 = boxes[ ..., 3] - (boxes[ ..., 3] - boxes[ ..., 1]) * a1
    quadrangles = tf.stack([quadrangles0, quadrangles1, quadrangles2, quadrangles3,
                     quadrangles4, quadrangles5, quadrangles6, quadrangles7], axis=-1)
    return quadrangles,boxes

def bbox_target_transform_inv(regression_target):
    boxes=regression_target[...,:4]
    a0 = regression_target[ ..., 4]
    a1 = regression_target[ ..., 5]
    quadrangles0 = boxes[ ..., 0] + (boxes[ ..., 2] - boxes[ ..., 0]) * a0
    quadrangles1 = boxes[ ..., 1]
    quadrangles2 = boxes[ ..., 2]
    quadrangles3 = boxes[ ..., 1] + (boxes[ ..., 3] - boxes[ ..., 1]) * a1
    quadrangles4 = boxes[ ..., 2] - (boxes[ ..., 2] - boxes[ ..., 0]) * a0
    quadrangles5 = boxes[ ..., 3]
    quadrangles6 = boxes[ ..., 0]
    quadrangles7 = boxes[ ..., 3] - (boxes[ ..., 3] - boxes[ ..., 1]) * a1
    return tf.stack([quadrangles0, quadrangles1, quadrangles2, quadrangles3,
                        quadrangles4, quadrangles5, quadrangles6, quadrangles7], axis=-1)



def focal(alpha=0.25, gamma=1.5):
    """
    Create a functor for computing the focal loss.
    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.
    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """

    def _focal(y_true, y_pred):
        """
        Compute the focal loss given the target tensor and the predicted tensor.
        As defined in https://arxiv.org/abs/1708.02002
        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).
        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        labels = y_true[:, :, :-1]
        # -1 for ignore, 0 for background, 1 for object
        anchor_state = y_true[:, :, -1]
        classification = y_pred

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.not_equal(anchor_state, -1))
        labels = tf.gather_nd(labels, indices)
        classification = tf.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = tf.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        # (1 - 0.99) ** 2 = 1e-4, (1 - 0.9) ** 2 = 1e-2
        focal_weight = tf.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma
        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        return keras.backend.sum(cls_loss) / normalizer

    return _focal


def smooth_l1(sigma=3.0):
    """
    Create a smooth L1 loss functor.
    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.
    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.
        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).
        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(regression_loss) / normalizer

    return _smooth_l1


def smooth_l1_quad(sigma=3.0):
    """
    Create a smooth L1 loss functor.
    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.
    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.
        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).
        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression = y_pred
        regression = tf.concat([regression[..., :4], tf.sigmoid(regression[..., 4:7])], axis=-1)
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        anchors = anchors_for_shape(image_size, anchor_params=AnchorParameters.default)
        # print('anchors.shape',anchors.shape)
        # anchors = tf.constant(anchors)
        # print('anchors.shape',anchors.shape)
        # print('regression_decode.shape',regression_decode.shape)
        # print('3')
        # regression_target_b = tf.gather_nd(regression_target, indices)
        # print('regression_target_b.shape',regression_target_b.shape)
        # print('4')
        # 
        # print('5')
        anchors = tf.gather(anchors, indices[:,1])

        regression_decode_quad,regression_decode_box = bbox_transform_inv(anchors,regression)
        # print('regression_decode_quad.shape',regression_decode_quad.shape)
        # print('regression_decode_box.shape',regression_decode_box.shape)

        
        box_regression_loss=1-bbox_overlaps_ciou(regression_decode_box,regression_target[...,:4])
        # print('box_regression_loss.shape',box_regression_loss.shape)
        regression_target_quad =  bbox_target_transform_inv(regression_target)
        # print('regression_target_quad.shape',regression_target_quad.shape)
        rt_box,rg_box = quad_to_off_bbox(regression_target_quad),quad_to_off_bbox(regression_decode_quad)
        
        quad_regression_loss=1-bbox_overlaps_ciou(rg_box,rt_box)
        # print('quad_regression_loss.shape',quad_regression_loss.shape)

        # print('6')
        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        # box_regression_loss = tf.where(
        #     keras.backend.less(regression_diff[..., :4], 1.0 / sigma_squared),
        #     0.5 * sigma_squared * keras.backend.pow(regression_diff[..., :4], 2),
        #     regression_diff[..., :4] - 0.5 / sigma_squared
        # )

        alpha_regression_loss = tf.where(
            keras.backend.less(regression_diff[..., 4:6], 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff[..., 4:6], 2),
            regression_diff[..., 4:6] - 0.5 / sigma_squared
        )

        ratio_regression_loss = tf.where(
            keras.backend.less(regression_diff[..., 6], 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff[..., 6], 2),
            regression_diff[..., 6] - 0.5 / sigma_squared
        )
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())

        box_regression_loss = tf.reduce_sum(box_regression_loss) / normalizer
        quad_regression_loss = tf.reduce_sum(quad_regression_loss) / normalizer

        alpha_regression_loss = tf.reduce_sum(alpha_regression_loss) / normalizer
        ratio_regression_loss = tf.reduce_sum(ratio_regression_loss) / normalizer

        return box_regression_loss  + quad_regression_loss + alpha_regression_loss +  ratio_regression_loss

    return _smooth_l1
