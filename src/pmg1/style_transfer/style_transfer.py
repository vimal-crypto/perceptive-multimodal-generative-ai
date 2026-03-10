import os
import numpy as np
import PIL.Image
from PIL import ImageEnhance
import tensorflow as tf

os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'


def tensor_to_image(tensor: tf.Tensor) -> PIL.Image.Image:
    """Convert a TensorFlow tensor (0-1 float) to a PIL Image."""
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img: str) -> tf.Tensor:
    """Load an image from disk and return as a normalized float32 tensor."""
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img[tf.newaxis, :]


def gram_matrix(input_tensor: tf.Tensor) -> tf.Tensor:
    """Compute the Gram matrix for style representation."""
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


def clip_0_1(image: tf.Tensor) -> tf.Tensor:
    """Clip tensor pixel values to [0, 1]."""
    return tf.clip_by_value(image, 0.0, 1.0)


def high_pass_x_y(image: tf.Tensor):
    """Compute high-frequency components for total variation loss."""
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
    return x_var, y_var


def vgg_layers(layer_names: list) -> tf.keras.Model:
    """
    Build a VGG19 feature extractor for the specified layer names.
    Uses ImageNet pre-trained weights with frozen parameters.
    """
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    return tf.keras.Model([vgg.input], outputs)


class StyleContentModel(tf.keras.models.Model):
    """
    A Keras model that extracts style (Gram matrices) and content features
    from VGG19 intermediate layers for neural style transfer.
    """

    def __init__(self, style_layers: list, content_layers: list):
        super().__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed)
        style_outputs = outputs[:self.num_style_layers]
        content_outputs = outputs[self.num_style_layers:]
        style_outputs = [gram_matrix(so) for so in style_outputs]
        content_dict = {name: val for name, val in zip(self.content_layers, content_outputs)}
        style_dict = {name: val for name, val in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}


def perform_style_transfer(
    content_image_path: str,
    style_image_path: str,
    output_path: str = "outputs/enhanced-stylized-image.png",
    epochs: int = 3,
    steps_per_epoch: int = 50,
    style_weight: float = 1e-2,
    content_weight: float = 1e4,
    total_variation_weight: float = 30.0,
    color_enhance_factor: float = 1.5
) -> str:
    """
    Perform neural style transfer using VGG19 feature matching.

    The content image is iteratively updated using Adam optimizer to minimize
    a combined style + content + total variation loss. Uses AdaIN-like
    feature normalization via Gram matrix style representation.

    Args:
        content_image_path: Path to the content image.
        style_image_path: Path to the style (artistic) image.
        output_path: Where to save the stylized output.
        epochs: Number of optimization epochs.
        steps_per_epoch: Gradient steps per epoch.
        style_weight: Weight for style loss term.
        content_weight: Weight for content loss term.
        total_variation_weight: Weight for total variation regularization.
        color_enhance_factor: Pillow color enhancement factor (1.0 = original).

    Returns:
        Path to the saved stylized image.
    """
    content_image = load_img(content_image_path)
    style_image = load_img(style_image_path)

    content_layers = ['block5_conv2']
    style_layers = [
        'block1_conv1', 'block2_conv1',
        'block3_conv1', 'block4_conv1', 'block5_conv1'
    ]

    extractor = StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    image = tf.Variable(content_image)
    opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    def style_content_loss(outputs):
        style_loss = tf.add_n([
            tf.reduce_mean((outputs['style'][n] - style_targets[n]) ** 2)
            for n in outputs['style']
        ]) * (style_weight / len(style_layers))
        content_loss = tf.add_n([
            tf.reduce_mean((outputs['content'][n] - content_targets[n]) ** 2)
            for n in outputs['content']
        ]) * (content_weight / len(content_layers))
        return style_loss + content_loss

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)
            loss += total_variation_weight * tf.image.total_variation(image)
        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    print(f"[INFO] Running style transfer: {epochs} epochs x {steps_per_epoch} steps")
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            train_step(image)
        print(f"[INFO] Epoch {epoch+1}/{epochs} complete.")

    result = tensor_to_image(image)
    result = ImageEnhance.Color(result).enhance(color_enhance_factor)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    result.save(output_path)
    print(f"[INFO] Stylized image saved: {output_path}")
    return output_path
