import os
import tensorflow as tf
import numpy as np
import PIL.Image
from PIL import ImageEnhance
import matplotlib.pyplot as plt

os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def perform_style_transfer(content_image_path, style_image_path):
    content_image = load_img(content_image_path)
    style_image = load_img(style_image_path)

    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

    def vgg_layers(layer_names):
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in layer_names]
        model = tf.keras.Model([vgg.input], outputs)
        return model

    class StyleContentModel(tf.keras.models.Model):
        def __init__(self, style_layers, content_layers):
            super(StyleContentModel, self).__init__()
            self.vgg = vgg_layers(style_layers + content_layers)
            self.style_layers = style_layers
            self.content_layers = content_layers
            self.num_style_layers = len(style_layers)
            self.vgg.trainable = False

        def call(self, inputs):
            inputs = inputs * 255.0
            preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
            outputs = self.vgg(preprocessed_input)
            style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
            style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
            content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
            style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
            return {'content': content_dict, 'style': style_dict}

    extractor = StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    image = tf.Variable(content_image)
    opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    style_weight = 1e-2
    content_weight = 1e4
    total_variation_weight = 30

    def style_content_loss(outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2) for name in style_outputs.keys()])
        style_loss *= style_weight / len(style_layers)
        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name])**2) for name in content_outputs.keys()])
        content_loss *= content_weight / len(content_layers)
        return style_loss + content_loss

    def high_pass_x_y(image):
        x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
        y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
        return x_var, y_var

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)
            loss += total_variation_weight * tf.image.total_variation(image)
        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    for n in range(1):
        for m in range(20):
            train_step(image)

    result_image = tensor_to_image(image)
    enhancer = ImageEnhance.Color(result_image)
    result_image = enhancer.enhance(1.5)
    enhanced_file_name = 'enhanced-stylized-image.png'
    result_image.save(enhanced_file_name)
    return enhanced_file_name
