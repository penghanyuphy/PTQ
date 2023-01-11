import paddle
import paddle.fluid as fluid
import paddle.dataset.mnist as reader
from paddleslim.models import MobileNet
from paddleslim.quant import quant_post_dynamic

= paddle.static.data(name='image', shape=[None, 1, 28, 28], dtype='float32')
paddle.static.save_inference_model(
    '/root/.paddlenlp/models/facebook/opt-1.3b'

