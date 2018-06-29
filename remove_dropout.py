#!/usr/bin/env python2

import argparse

import tensorflow as tf
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2

def print_graph(input_graph):
    for node in input_graph.node:
        print "{0} : {1} ( {2} )".format(node.name, node.op, node.input)

def strip(input_graph, drop_scope, input_before, output_after, pl_name):
    input_nodes = input_graph.node
    nodes_after_strip = []
    for node in input_nodes:
        #print "{0} : {1} ( {2} )".format(node.name, node.op, node.input)

        if node.name.startswith(drop_scope + '/'):
            print "Remove:"
            print "{0} : {1} ( {2} )".format(node.name, node.op, node.input)
            continue

        if node.name == pl_name:
            continue

        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        if new_node.name.startswith(output_after + '/'):
            new_input = []
            for node_name in new_node.input:
                #print node_name
                if node_name == drop_scope + '/cond/Merge':
                    new_input.append(input_before)
                    #ALTER THE INPUT --yingjiun
                    print"remove /cond/Merge append",input_before
                    
                else:
                    new_input.append(node_name)
            del new_node.input[:]
            new_node.input.extend(new_input)
        nodes_after_strip.append(new_node)

    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(nodes_after_strip)
    return output_graph

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-graph', action='store', dest='input_graph')
    parser.add_argument('--input-binary', action='store_true', default=True, dest='input_binary')
    parser.add_argument('--output-graph', action='store', dest='output_graph')
    parser.add_argument('--output-binary', action='store_true', dest='output_binary', default=True)
    parser.add_argument('--drop-scope', action='store', dest='drop_scope')
    parser.add_argument('--input-before', action='store', dest='input_before')
    parser.add_argument('--output-after', action='store', dest='output_after')
    
    
    args = parser.parse_args()

    input_graph = args.input_graph
    input_binary = args.input_binary
    output_graph = args.output_graph
    output_binary = args.output_binary
    drop_scope = args.drop_scope
    input_before = args.input_before
    output_after = args.output_after
    

    if not tf.gfile.Exists(input_graph):
        print("Input graph file '" + input_graph + "' does not exist!")
        return

    input_graph_def = tf.GraphDef()
    mode = "rb" if input_binary else "r"
    with tf.gfile.FastGFile(input_graph, mode) as f:
        if input_binary:
            input_graph_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read().decode("utf-8"), input_graph_def)

    #print "Before:"
    #print_graph(input_graph_def)
    
    #CRUCIAL STEP, RUN THIS STEP BY STEP--yingjiun
    #output_graph_def = strip(input_graph_def, u'dropout_1', u'lstm_1/transpose_1', u'lstm_2/transpose', u'is_training_pl')
    #output_graph_def = strip(input_graph_def, u'dropout_2', u'lstm_2/TensorArrayReadV3', u'dense_1/MatMul', u'is_training_pl')
    
    #Generalize
    output_graph_def = strip(input_graph_def, drop_scope, input_before, output_after, u'is_training_pl')
    
    
    #print "After:"
    #print_graph(output_graph_def)

    if output_binary:
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
    else:
        with tf.gfile.GFile(output_graph, "w") as f:
            f.write(text_format.MessageToString(output_graph_def))
    print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == "__main__":
    main()#!/usr/b