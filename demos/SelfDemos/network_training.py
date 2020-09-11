# coding=utf-8
from phi.tf.flow import *

from tensorflow import keras
from tensorflow.keras import layers


DOMAIN = Domain([64, 64], boundaries=OPEN)  # [y, x]
DATAPATH = os.path.expanduser('~/phi/data/smoke/')  # at least 10 sims, has to match RESOLUTION
DESCRIPTION = u"""
Train a neural network to reproduce the flow field given the marker density.

This application loads the previously generated training data from "%s".

Try showing only the X component of the velocity - that one is more interesting than the magnitude.
Use the batch slider to change the example shown in the viewer.

Also make sure to check out the TensorBoard integration.
In TensorBoard you will see how the loss changes during optimization.
You can launch TensorBord right from the GUI by opening the Φ Board page and clicking on 'Launch TensorBoard'.
""" % DATAPATH


if not os.path.exists(os.path.join(DATAPATH, 'sim_000009')):
    print('Not enough training data found in %s. Run smoke_datagen_commandline.py or smoke_datagen_interactive.py to generate training data.' % DATAPATH)
    exit(1)

def conv(input, kernel, input_channels, output_channels):
    f1 = tf.nn.conv2d(input,
                      filters=[kernel, kernel, input_channels, output_channels],
                      strides=1,
                      padding="SAME")
    f2 = tf.nn.max_pool2d(f1,
                          ksize=kernel,
                          strides=kernel)
    f3 = tf.nn.leaky_relu(f2)
    return f3

class TrainingTest(LearningApp):

    def __init__(self):
        LearningApp.__init__(self, 'Training', DESCRIPTION, learning_rate=2e-4, validation_batch_size=4, training_batch_size=8)
        # --- Setup simulation and placeholders ---
        fluid_placeholders, load_dict = build_graph_input(Fluid(DOMAIN), input_type='dataset_handle')
        # --- Build neural network ---
        with self.model_scope():
            pred_vel = network(fluid_placeholders.density.data)
        # --- Loss function ---
        target_vel = fluid_placeholders.velocity.staggered_tensor()[..., :-1, :-1, :]
        loss = math.l2_loss(pred_vel - target_vel)
        self.add_objective(loss, 'Loss')
        # --- Training data ---
        self.set_data(dict=load_dict,
                      train=Dataset.load(DATAPATH, range(0, 8)),
                      val=Dataset.load(DATAPATH, range(8, 10)))
        # --- GUI ---
        self.add_field('Velocity (Ground Truth)', fluid_placeholders.velocity)
        self.add_field('Velocity (Model)', pred_vel)
        self.add_field('Density (Input)', fluid_placeholders.density)


show(display=('Velocity (Ground Truth)', 'Velocity (Model)'))
