import argparse
import os

import onnx
import torch
import torch.onnx

from models import dist_model as dm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['net-lin', 'net'], default='net-lin', help='net-lin or net')
    parser.add_argument('--net', choices=['squeeze', 'alex', 'vgg'], default='alex', help='squeeze, alex, or vgg')
    parser.add_argument('--version', type=str, default='0.1')
    parser.add_argument('--image_height', type=int, default=64)
    parser.add_argument('--image_width', type=int, default=64)
    args = parser.parse_args()

    model = dm.DistModel()
    model.initialize(model=args.model, net=args.net, use_gpu=False, version=args.version)
    print('Model [%s] initialized' % model.name())

    dummy_im0 = torch.Tensor(1, 3, args.image_height, args.image_width)  # image should be RGB, normalized to [-1, 1]
    dummy_im1 = torch.Tensor(1, 3, args.image_height, args.image_width)

    cache_dir = os.path.expanduser('~/.lpips')
    os.makedirs(cache_dir, exist_ok=True)
    onnx_fname = os.path.join(cache_dir, '%s_%s_v%s.onnx' % (args.model, args.net, args.version))

    # export model to onnx format
    torch.onnx.export(model.net, (dummy_im0, dummy_im1), onnx_fname, verbose=True)

    # load and change dimensions to be dynamic
    model = onnx.load(onnx_fname)
    for dim in (0, 2, 3):
        model.graph.input[0].type.tensor_type.shape.dim[dim].dim_param = '?'
        model.graph.input[1].type.tensor_type.shape.dim[dim].dim_param = '?'

    # needs to be imported after all the pytorch stuff, otherwise this causes a segfault
    from onnx_tf.backend import prepare
    tf_rep = prepare(model, device='CPU')
    producer_version = tf_rep.graph.graph_def_versions.producer
    pb_fname = os.path.join(cache_dir, '%s_%s_v%s_%d.pb' % (args.model, args.net, args.version, producer_version))
    tf_rep.export_graph(pb_fname)
    input0_name, input1_name = [tf_rep.tensor_dict[input_name].name for input_name in tf_rep.inputs]
    (output_name,) = [tf_rep.tensor_dict[output_name].name for output_name in tf_rep.outputs]

    # ensure these are the names of the 2 inputs, since that will be assumed when loading the pb file
    assert input0_name == '0:0'
    assert input1_name == '1:0'
    # ensure that the only output is the output of the last op in the graph, since that will be assumed later
    (last_output_name,) = [output.name for output in tf_rep.graph.get_operations()[-1].outputs]
    assert output_name == last_output_name


if __name__ == '__main__':
    main()
