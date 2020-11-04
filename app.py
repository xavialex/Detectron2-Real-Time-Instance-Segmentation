from pathlib import Path
import configparser
import cv2
import numpy as np
import threading
import video_utils
import sys
import streamlit as st

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


#@st.cache
def video_stream(video_source):
    #video_source = available_cameras[cam_id]
    video_thread = video_utils.WebcamVideoStream(video_source)
    video_thread.start()

    return video_thread

@st.cache(persist=True)
def initialization():
    """Loads configuration and model for the prediction.
    
    Returns:
        cfg (detectron2.config.config.CfgNode): Configuration for the model.
        classes_names (list: str): Classes available for the model of interest.
        predictor (detectron2.engine.defaults.DefaultPredicto): Model to use.
            by the model.
        
    """
    cfg = get_cfg()
    # Force model to operate within CPU, erase if CUDA compatible devices ara available
    cfg.MODEL.DEVICE = 'cpu'
    # Add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    # Set threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    # Get classes names for the dataset of interest
    classes_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    # Initialize prediction model
    predictor = DefaultPredictor(cfg)

    return cfg, classes_names, predictor


def inference(predictor, img):
    return predictor(img)


@st.cache
def output_image(cfg, img, outputs):
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    processed_img = out.get_image() #[:, :, ::-1]

    return processed_img


@st.cache
def discriminate(outputs, classes_to_detect):
    """Select which classes to detect from an output.

    Get the dictionary associated with the outputs instances and modify
    it according to the given classes to restrict the detection to them

    Args:
        outputs (dict):
            instances (detectron2.structures.instances.Instances): Instance
                element which contains, among others, "pred_boxes", 
                "pred_classes", "scores" and "pred_masks".
        classes_to_detect (list: int): Identifiers of the dataset on which
            the model was trained.

    Returns:
        ouputs (dict): Same dict as before, but modified to match
            the detection classes.

    """
    print('aaaaa')
    print(outputs['instances'].pred_classes)
    pred_classes = np.array(outputs['instances'].pred_classes)
    # Take the elements matching *classes_to_detect*
    mask = np.isin(pred_classes, classes_to_detect)
    # Get the indexes
    idx = np.nonzero(mask)

    # # Get the current Instance values
    # pred_boxes = outputs['instances'].pred_boxes
    # pred_classes = outputs['instances'].pred_classes
    # pred_masks = outputs['instances'].pred_masks
    # scores = outputs['instances'].scores

    # Get Instance values as a dict and leave only the desired ones
    out_fields = outputs['instances'].get_fields()
    for field in out_fields:
        out_fields[field] = out_fields[field][idx]

    return outputs








def main():
    # Initialization
    cfg, classes, predictor = initialization()

    # Streamlit initialization
    st.title("Instance Segmentation")
    st.sidebar.title("Options")
    ## Select classes to be detected by the model
    classes_to_detect = st.sidebar.multiselect(
        "Select which classes to detect", classes, ['person'])
    mask = np.isin(classes, classes_to_detect)
    class_idxs = np.nonzero(mask)
    ## Select camera to feed the model
    available_cameras = {'Camera 1': 0, 'Camera 2': 1, 'Camera 3': 2}
    cam_id = st.sidebar.selectbox(
        "Select which camera signal to use", list(available_cameras.keys()))

    # Define holder for the processed image
    img_placeholder = st.empty()

    # Load video source into a thread
    
    video_source = available_cameras[cam_id]
    video_thread = video_stream(video_source)
    # video_thread = video_utils.WebcamVideoStream(video_source)
    # video_thread.start()
    
    # Detection code
    try:
        while not video_thread.stopped():
            # Camera detection loop
            frame = video_thread.read()
            if frame is None:
                print("Frame stream interrupted")
                break
            # Change color gammut to feed the frame into the network
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Detection code
            outputs = inference(predictor, frame)
            outputs = discriminate(outputs, class_idxs)
            out_image = output_image(cfg, frame, outputs)
            #st.image(out_image, caption='Processed Image', use_column_width=True)        
            img_placeholder.image(out_image)





            # output = run_inference_for_single_image(frame, sess, 
            #     detection_graph)
            # output = discriminate_class(output, 
            #     classes_to_detect, category_index)
            # processed_image = visualize_results(frame, output, 
            #     category_index)

            # # Display the image with the detections in the Streamlit app
            # img_placeholder.image(processed_image)
            
            # #cv2.imshow('Video', cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

            # # if cv2.waitKey(1) & 0xFF == ord('q'):
            # #     break
    
    except KeyboardInterrupt:   
        pass

    print("Ending resources")
    st.text("Camera not detected")
    cv2.destroyAllWindows()
    video_thread.stop()
    sys.exit()












    # # Initialization
    # ## Load the configuration variables from 'config.ini'
    # config = configparser.ConfigParser()
    # config.read('config.ini')
    # ## Loading label map
    # num_classes = config.getint('net', 'num_classes')
    # path_to_labels = config['net']['path_to_labels']
    # label_map = label_map_util.load_labelmap(path_to_labels)
    # categories = label_map_util.convert_label_map_to_categories(label_map, 
    #     max_num_classes=num_classes, use_display_name=True)
    # category_index = label_map_util.create_category_index(categories)

    # # Streamlit initialization
    # st.title("Object Detection")
    # st.sidebar.title("Object Detection")
    # ## Select classes to be detected by the model
    # classes_names = [value['name'] for value in category_index.values()]
    # classes_names.sort()
    # classes_to_detect = st.sidebar.multiselect(
    #     "Select which classes to detect", classes_names, ['person'])
    # ## Select camera to feed the model
    # available_cameras = {'Camera 1': 0, 'Camera 2': 1, 'Camera 3': 2}
    # cam_id = st.sidebar.selectbox(
    #     "Select which camera signal to use", list(available_cameras.keys()))
    # ## Select a model to perform the inference
    # available_models = [str(i) for i in Path('./trained_model/').iterdir() 
    #     if i.is_dir() and list(Path(i).glob('*.pb'))]
    # model_name = st.sidebar.selectbox(
    #     "Select which model to use", available_models)
    # # Define holder for the processed image
    # img_placeholder = st.empty()

    # # Model load
    # path_to_ckpt = '{}/frozen_inference_graph.pb'.format(model_name)
    # detection_graph = model_load_into_memory(path_to_ckpt)

    # # Load video source into a thread
    # video_source = available_cameras[cam_id]
    # ## Start video thread
    # video_thread = video_utils.WebcamVideoStream(video_source)
    # video_thread.start()
    
    # # Detection code
    # try:
    #     with detection_graph.as_default():
    #         with tf.Session(graph=detection_graph) as sess:
    #             while not video_thread.stopped():
    #                 # Camera detection loop
    #                 frame = video_thread.read()
    #                 if frame is None:
    #                     print("Frame stream interrupted")
    #                     break
    #                 # Change color gammut to feed the frame into the network
    #                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #                 output = run_inference_for_single_image(frame, sess, 
    #                     detection_graph)
    #                 output = discriminate_class(output, 
    #                     classes_to_detect, category_index)
    #                 processed_image = visualize_results(frame, output, 
    #                     category_index)

    #                 # Display the image with the detections in the Streamlit app
    #                 img_placeholder.image(processed_image)
                    
    #                 #cv2.imshow('Video', cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

    #                 # if cv2.waitKey(1) & 0xFF == ord('q'):
    #                 #     break
    
    # except KeyboardInterrupt:   
    #     pass

    # print("Ending resources")
    # st.text("Camera not detected")
    # cv2.destroyAllWindows()
    # video_thread.stop()
    # sys.exit()


if __name__ == '__main__':
    main()
    