#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers import LeakyReLU, BatchNormalization, Activation, Dropout, add, multiply, maximum, average
import keras.backend as K
from my_merge import seven_three, three_seven, nine_one, six_four, eight_two


class XNet(object):
    def __init__(self, input_channel_count, output_channel_count, first_layer_filter_count):
        self.INPUT_IMAGE_SIZE = 256
        self.CONCATENATE_AXIS = -1
        self.CONV_FILTER_SIZE = 4
        self.CONV_STRIDE = 2
        self.CONV_PADDING = (1, 1)
        self.DECONV_FILTER_SIZE = 2
        self.DECONV_STRIDE = 2
        
        # (256 x 256 x input_channel_count)  64
        inputs = Input((self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))  # ordinary
        output_ordinary, a, b, c, d, e, f, g, filter_count_enc = self._encoder(inputs, first_layer_filter_count)
        inputs_R = Input((self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))
        output_reverse, A, B, C, D, E, F, G, filter_count_enc = self._encoder(inputs_R, first_layer_filter_count)

        added_o = seven_three([output_ordinary, output_reverse])
        added_r = seven_three([output_reverse, output_ordinary])

        sk_A = add([a,A])
        sk_B = add([b,B])
        sk_C = add([c, C])
        sk_D = add([d, D])
        sk_E = add([e, E])
        sk_F = add([f, F])
        sk_G = add([g, G])

        dec7_o = self._decoder(added_o, sk_A, sk_B, sk_C, sk_D, sk_E, sk_F, sk_G, filter_count_enc, output_channel_count,
                             first_layer_filter_count)

        dec7_r = self._decoder(added_r, A,B,C,D,E,F,G, filter_count_enc, output_channel_count,
                               first_layer_filter_count)

        # (256 x 256 x output_channel_count)64
        dec8_o = Activation(activation='relu')(dec7_o)
        dec8_o = Conv2DTranspose(output_channel_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE)(dec8_o)
        dec8_o = Activation(activation='sigmoid')(dec8_o)  # comment this! for losvas loss

        # (256 x 256 x output_channel_count)64
        dec8_r = Activation(activation='relu')(dec7_r)
        dec8_r = Conv2DTranspose(output_channel_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE)(dec8_r)
        dec8_r = Activation(activation='sigmoid')(dec8_r)  # comment this! for losvas loss

        dec8 = maximum([dec8_o,dec8_r])
        self.YNET = Model(input=[inputs,inputs_R], output=dec8)

    def _encoder(self, inputs, first_layer_filter_count):
        # Encoder part: エンコーダーの作成
        # (128 x 128 x N)32
        enc1 = ZeroPadding2D(self.CONV_PADDING)(inputs)
        enc1 = Conv2D(first_layer_filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(enc1)

        # (64 x 64 x 2N)16
        filter_count = first_layer_filter_count * 2
        enc2 = self._add_encoding_layer(filter_count, enc1)

        # (32 x 32 x 4N)8
        filter_count = first_layer_filter_count * 4
        enc3 = self._add_encoding_layer(filter_count, enc2)

        # (16 x 16 x 8N)4
        filter_count = first_layer_filter_count * 8
        enc4 = self._add_encoding_layer(filter_count, enc3)

        # (8 x 8 x 8N)2
        enc5 = self._add_encoding_layer(filter_count, enc4)

        # (4 x 4 x 8N)1
        enc6 = self._add_encoding_layer(filter_count, enc5)

        # (2 x 2 x 8N)
        enc7 = self._add_encoding_layer(filter_count, enc6)

        # (1 x 1 x 8N)
        enc8 = self._add_encoding_layer(filter_count, enc7)

        return enc8, enc7,enc6, enc5, enc4, enc3, enc2, enc1, filter_count

    def _reverseinputlayer(self,input):
        temp = input
        inputs_R0 = temp[:, :, :, 1]
        inputs_R1 = temp[:, :, :, 0]
        inputs_R0 = K.expand_dims(inputs_R0, axis=-1)
        inputs_R1 = K.expand_dims(inputs_R1, axis=-1)
        inputs_R = concatenate([inputs_R0, inputs_R1], axis=-1)
        return inputs_R

    def _decoder(self, enc8, enc7, enc6, enc5, enc4, enc3, enc2, enc1, filter_count, output_channel_count,first_layer_filter_count):

        dec1 = self._add_decoding_layer(filter_count, True, enc8)
        dec1 = concatenate([dec1, enc7], axis=self.CONCATENATE_AXIS)

        # (4 x 4 x 8N)
        dec2 = self._add_decoding_layer(filter_count, True, dec1)
        dec2 = concatenate([dec2, enc6], axis=self.CONCATENATE_AXIS)

        # (8 x 8 x 8N)2
        dec3 = self._add_decoding_layer(filter_count, True, dec2)
        dec3 = concatenate([dec3, enc5], axis=self.CONCATENATE_AXIS)

        # (16 x 16 x 8N)4
        dec4 = self._add_decoding_layer(filter_count, False, dec3)
        dec4 = concatenate([dec4, enc4], axis=self.CONCATENATE_AXIS)

        # (32 x 32 x 4N)8
        filter_count = first_layer_filter_count * 4
        dec5 = self._add_decoding_layer(filter_count, False, dec4)
        dec5 = concatenate([dec5, enc3], axis=self.CONCATENATE_AXIS)

        # (64 x 64 x 2N)16
        filter_count = first_layer_filter_count * 2
        dec6 = self._add_decoding_layer(filter_count, False, dec5)
        dec6 = concatenate([dec6, enc2], axis=self.CONCATENATE_AXIS)

        # (128 x 128 x N)32
        filter_count = first_layer_filter_count
        dec7 = self._add_decoding_layer(filter_count, False, dec6)
        dec7 = concatenate([dec7, enc1], axis=self.CONCATENATE_AXIS)

        # (256 x 256 x output_channel_count)64
        # dec8 = Activation(activation='relu')(dec7)
        # dec8 = Conv2DTranspose(output_channel_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE)(dec8)
        # dec8 = Activation(activation='sigmoid')(dec8)  # comment this! for losvas loss

        return dec7#dec8

    def _add_encoding_layer(self, filter_count, sequence):
        new_sequence = LeakyReLU(0.2)(sequence)
        new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
        new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        return new_sequence

    def _add_decoding_layer(self, filter_count, add_drop_layer, sequence):
        new_sequence = Activation(activation='relu')(sequence)
        new_sequence = Conv2DTranspose(filter_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE,
                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        if add_drop_layer:
            new_sequence = Dropout(0.5)(new_sequence)
        return new_sequence

    def get_model(self):
        return self.YNET





