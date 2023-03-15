dutils.img_save(tensor_to_numpy(self.y_pyramid[0][0,...,:3]),'y_pyramid.png')
dutils.img_save(tensor_to_numpy(self.y_pyramid[-2][0,...,:3]),'y_pyramid.png')
INPAINTING_HIGH_SCALE=1 python -m benchmark.run_metrics  --methodnames gradcam --metrics gpnn_eval --modelname vgg16 --delete_percentiles 80
