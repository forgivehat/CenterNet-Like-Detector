import sys
import threading
from mmdet.apis import init_detector, inference_detector

def inference(model, image_filename, output_filename):
    result = inference_detector(model, f'/data/SODA-D/rawData/Images/{image_filename}.jpg')
    model.show_result(f'/data/SODA-D/rawData/Images/{image_filename}.jpg', result, score_thr=0.3, show=False, out_file=output_filename)

def init_model(device, config_file, checkpoint_file):
    return init_detector(config_file, checkpoint_file, device=device)

def main(start_image_number):
    device1 = 'cpu'
    config_file1 = 'configs/cfinet/faster_rcnn_r50_fpn_cfinet_1x.py'
    checkpoint_file1 = 'work_dirs/faster_rcnn_r50_fpn_cfinet_1x/epoch_12.pth'
    model1 = init_model(device1, config_file1, checkpoint_file1)

    device2 = 'cpu'
    config_file2 = 'configs/centernet_like/centernet_r50_repeat_dataset_70e_ulike_spp_dcnv2_attention.py'
    checkpoint_file2 = 'work_dirs/centernet_r50_repeat_dataset_70e_ulike_spp_dcnv2_attention/latest.pth'
    model2 = init_model(device2, config_file2, checkpoint_file2)

    for i in range(start_image_number, start_image_number + 100):
        image_filename = f"{i:04d}"
        output_filename1 = f'../result_2/{image_filename}_result_cfi.jpg'
        output_filename2 = f'../result_2/{image_filename}_result_ct.jpg'

        # 创建并启动两个线程进行推理
        thread1 = threading.Thread(target=inference, args=(model1, image_filename, output_filename1))
        thread2 = threading.Thread(target=inference, args=(model2, image_filename, output_filename2))
        thread1.start()
        thread2.start()

        # 等待两个线程完成
        thread1.join()
        thread2.join()

if __name__ == "__main__":
    # 随便选
    start_image_number = 13200 
    main(start_image_number)
