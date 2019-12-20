import paddle.fluid as fluid

from config_init_utils import init_train_parameters
from config_init_utils import train_parameters as train_parameters
from yolo_utils import get_yolo


init_train_parameters()


def freeze_model():
    exe = fluid.Executor(fluid.CPUPlace())
    ues_tiny = train_parameters['use_tiny']
    yolo_config = train_parameters['yolo_tiny_cfg'] if ues_tiny else train_parameters['yolo_cfg']
    model_path = train_parameters['save_model_dir']
    model = get_yolo(ues_tiny,
                     train_parameters['class_dim'],
                     yolo_config['anchors'],
                     yolo_config['anchor_mask'])
    input_image_shape = fluid.layers.data(name='input_image_shape', shape=yolo_config['input_size'], dtype='float32')
    image_shape = fluid.layers.data(name="image_shape", shape=[2], dtype='int32')

    boxes = []
    scores = []
    outputs = model.net(input_image_shape)
    downsample_ratio = model.get_downsample_ratio()
    for i, out in enumerate(outputs):
        box, score = fluid.layers.yolo_box(
            x=out,
            img_size=image_shape,
            anchors=model.get_yolo_anchors()[i],
            class_num=model.get_class_num(),
            conf_thresh=train_parameters['valid_thresh'],
            downsample_ratio=downsample_ratio,
            name="yolo_box_" + str(i))
        boxes.append(box)
        scores.append(fluid.layers.transpose(score, perm=[0, 2, 1]))
        downsample_ratio //= 2

    pred = fluid.layers.multiclass_nms(
        bboxes=fluid.layers.concat(boxes, axis=1),
        scores=fluid.layers.concat(scores, axis=2),
        score_threshold=train_parameters['valid_thresh'],
        nms_top_k=train_parameters['nms_top_k'],
        keep_top_k=train_parameters['nms_pos_k'],
        nms_threshold=train_parameters['nms_thresh'],
        background_label=-1,
        name="multiclass_nms")

    freeze_program = fluid.default_main_program()
    fluid.io.load_persistables(exe, model_path, freeze_program)
    freeze_program = freeze_program.clone(for_test=True)
    print("freeze out dir: {}".format(train_parameters['freeze_dir']))
    print("pred layout:\n{}".format(pred))
    fluid.io.save_inference_model(train_parameters['freeze_dir'],
                                  ['input_image_shape', 'image_shape'],
                                  pred,
                                  exe,
                                  freeze_program)
    print("freeze end")


if __name__ == '__main__':
    freeze_model()
