## Quick Start

* [下载数据集](https://challenger.ai/competition/scene/subject)


* 配置数据集路径

  打开`config.py`，找到下面的位置，根据自己的电脑系统在对应的位置配置上数据集路径

  ```
  # image path
  if is_windows():
      PATH_TRAIN_BASE = 'G:/Dataset/SceneClassify/ai_challenger_scene_train_20170904'
      PATH_VAL_BASE = 'G:/Dataset/SceneClassify/ai_challenger_scene_validation_20170908'
      PATH_TEST_B = 'G:/Dataset/SceneClassify/ai_challenger_scene_test_b_20170922/scene_test_b_images_20170922'
  elif is_mac():
      PATH_TRAIN_BASE = '/Users/zijiao/Desktop/ai_challenger_scene_train_20170904'
      PATH_VAL_BASE = '/Users/zijiao/Desktop/ai_challenger_scene_validation_20170908'
      PATH_TEST_B = ''
  elif is_linux():
      # 皮皮酱
      PATH_TRAIN_BASE = ''
      PATH_VAL_BASE = ''
      PATH_TEST_B = ''
  else:
      raise Exception('No images configured on %s' % os_name)
  ```

* 分类数据

  运行`split_by_class.py ` 脚本，分别对train数据集合val数据集进行按照子文件夹分类

* 开始训练

  找任一个`classifier` 开头的(`classifier_base` 除外)脚本进行运行，这里包含`VGG16/19`、`Xception`、`Inception-V3`、`Inception-Resnet-V2`等经典模型

