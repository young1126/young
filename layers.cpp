#include "layers.h"

// --------------------------------------------------------
// 1) AXI Stream 유틸리티
// --------------------------------------------------------
void load_input(D_TYPE *target, AXI_VAL *source) {
    for (int i = 0; i < BATCH_SIZE * 25; i++) {
        target[i] = pop_stream_uint32(source[i]);
    }
}

void store_output(AXI_VAL *target, FC_TYPE *source) {
    for (int i = 0; i < BATCH_SIZE * FC_OUTPUT - 1; i++) {
        uint32_t value = (uint32_t)(source[i].to_uint() & 0xFFFF);
        target[i] = push_stream_uint32(value, false);
    }
    uint32_t last_value = (uint32_t)(source[BATCH_SIZE * FC_OUTPUT - 1].to_uint() & 0xFFFF);
    target[BATCH_SIZE * FC_OUTPUT - 1] = push_stream_uint32(last_value, true);
}

// --------------------------------------------------------
// 2) 이진 언패킹 함수 - MSB부터 언패킹하도록 수정
// --------------------------------------------------------
void unpack_binary_input(D_TYPE input_data[25], ap_uint<1> output_data[IMG_H*IMG_W]) {
    for (int word_idx = 0; word_idx < 25; word_idx++) {
        D_TYPE packed_word = input_data[word_idx];
        for (int bit_idx = 0; bit_idx < 32; bit_idx++) {
            int global_pixel_idx = word_idx * 32 + bit_idx;
            if (global_pixel_idx < IMG_H*IMG_W) {
                output_data[global_pixel_idx] = (packed_word >> (31 - bit_idx)) & 1;
            }
        }
    }
}

// --------------------------------------------------------
// 3) 유틸리티 함수
// --------------------------------------------------------
inline ap_uint<1> binarize(ap_int<8> x) { return (x >= 0) ? 1 : 0; }
inline ap_int<8> relu(ap_int<16> x) {
    #pragma HLS INLINE
    return (x > 0) ? (ap_int<8>)x : (ap_int<8>)0;
}

inline ap_int<8> bit_to_pm1(ap_uint<1> bit) {
    #pragma HLS INLINE
    return bit ? (ap_int<8>)1 : (ap_int<8>)-1;
}

const char pop_count_table[256] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

inline ap_int<8> popcount32_lookup(ap_uint<32> x) {
    #pragma HLS INLINE
    #pragma HLS RESOURCE variable=pop_count_table core=ROM_1P
    ap_int<8> res = 0;
    res += pop_count_table[(x >> 24) & 0xFF];
    res += pop_count_table[(x >> 16) & 0xFF];
    res += pop_count_table[(x >> 8)  & 0xFF];
    res += pop_count_table[x & 0xFF];
    return res;
}

inline ap_int<8> xnor_popcount(ap_uint<32> input, ap_uint<32> weight) {
    #pragma HLS INLINE
    ap_uint<32> xnor_result = ~(input ^ weight);
    ap_int<8> popcount = popcount32_lookup(xnor_result);
    return (popcount << 1) - 32;
}

inline ap_int<8> xnor_popcount9(ap_uint<32> in, ap_uint<32> wt) {
    #pragma HLS INLINE
    const ap_uint<32> mask = 0x1FF;
    ap_uint<32> xnor_res = ~(in ^ wt) & mask;
    ap_int<8> pc = popcount32_lookup(xnor_res);
    return (pc << 1) - 9;
}

// --------------------------------------------------------
// 4) Conv1 Layer
// --------------------------------------------------------
void conv1_layer(
    ap_uint<1> input_data[IMG_H * IMG_W],
    ap_int<8> output_data[CONV1_OUT_H * CONV1_OUT_W * CONV1_OUT_C]
) {
    #pragma HLS INLINE off

    ap_uint<1> line_buffer[3][IMG_W + 2];
    #pragma HLS ARRAY_PARTITION variable=line_buffer dim=1 complete
    #pragma HLS RESOURCE variable=conv1_weights core=ROM_1P

    ap_uint<1> window[3][3];
    #pragma HLS ARRAY_PARTITION variable=window complete

    // 라인 버퍼 초기화
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < IMG_W + 2; c++)
            line_buffer[r][c] = 0;

    int out_idx = 0;
    for (int row = 0; row < IMG_H; row++) {
        for (int col = 0; col < IMG_W; col++) {
            #pragma HLS PIPELINE
            ap_uint<1> curr = input_data[row * IMG_W + col];

            // line buffer shift
            line_buffer[2][col + 1] = line_buffer[1][col + 1];
            line_buffer[1][col + 1] = line_buffer[0][col + 1];
            line_buffer[0][col + 1] = curr;

            // window buffer shift
            for (int r = 0; r < 3; r++) {
                window[r][0] = window[r][1];
                window[r][1] = window[r][2];
                window[r][2] = line_buffer[r][col + 1];
            }

            if (row >= 2 && col >= 2) {
                for (int ch = 0; ch < CONV1_OUT_C; ch++) {
                    ap_int<16> sum = 0;
                    for (int kr = 0; kr < 3; kr++) {
                        for (int kc = 0; kc < 3; kc++) {
                            ap_int<8> pix = bit_to_pm1(window[kr][kc]);
                            sum += conv1_weights[ch][0][kr][kc] * pix;
                        }
                    }
                    // sign 함수 구현
                    ap_int<8> out_val;
                    if      (sum >  0) out_val =  1;
                    else if (sum <  0) out_val = -1;
                    else               out_val =  0;
                    output_data[out_idx + ch] = out_val;
                }
                out_idx += CONV1_OUT_C;
            }
        }
    }
}

// --------------------------------------------------------
// 5) Conv2 Layer
// --------------------------------------------------------
void conv2_layer(
    ap_int<8> input_data[CONV1_OUT_H * CONV1_OUT_W * CONV1_OUT_C],
    ap_int<8> output_data[CONV2_OUT_H * CONV2_OUT_W * CONV2_OUT_C]
) {
    #pragma HLS INLINE off

    ap_uint<1> line_buffer[CONV2_IN_C][3][CONV1_OUT_W + 2];
    #pragma HLS ARRAY_PARTITION variable=line_buffer dim=1 cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=line_buffer dim=2 complete
    #pragma HLS RESOURCE variable=line_buffer core=RAM_1P_BRAM

    ap_uint<1> window[CONV2_IN_C][3][3];
    #pragma HLS ARRAY_PARTITION variable=window dim=1 cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=window dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=window dim=3 complete

    #pragma HLS RESOURCE variable=conv2_weights core=ROM_1P
    #pragma HLS ARRAY_PARTITION variable=conv2_weights dim=1 cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=conv2_weights dim=2 cyclic factor=4

    // 버퍼 초기화
    for (int ch = 0; ch < CONV2_IN_C; ch++) {
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < CONV1_OUT_W + 2; c++) {
                line_buffer[ch][r][c] = 0;
            }
        }
    }

    int out_idx = 0;
    for (int row = 0; row < CONV1_OUT_H; row++) {
        for (int col = 0; col < CONV1_OUT_W; col++) {
            // 입력 채널을 4개씩 묶어서 처리
            for (int ch_base = 0; ch_base < CONV2_IN_C; ch_base += 4) {
                #pragma HLS PIPELINE II=1

                for (int ch_off = 0; ch_off < 4; ch_off++) {
                    #pragma HLS UNROLL
                    int ch = ch_base + ch_off;
                    if (ch < CONV2_IN_C) {
                        ap_int<8> pixel = input_data[(row * CONV1_OUT_W + col) * CONV2_IN_C + ch];
                        ap_uint<1> bit = (pixel >= 0) ? 1 : 0;

                        // 라인버퍼 업데이트
                        line_buffer[ch][2][col + 1] = line_buffer[ch][1][col + 1];
                        line_buffer[ch][1][col + 1] = line_buffer[ch][0][col + 1];
                        line_buffer[ch][0][col + 1] = bit;

                        // 윈도우 업데이트
                        for (int r = 0; r < 3; r++) {
                            #pragma HLS UNROLL
                            window[ch][r][0] = window[ch][r][1];
                            window[ch][r][1] = window[ch][r][2];
                            window[ch][r][2] = line_buffer[ch][r][col + 1];
                        }
                    }
                }
            }

            // 컨볼루션 계산
            if (row >= 2 && col >= 2) {
                for (int och_base = 0; och_base < CONV2_OUT_C; och_base += 2) {
                    #pragma HLS PIPELINE II=1

                    ap_int<16> sums[2] = {0, 0};
                    #pragma HLS ARRAY_PARTITION variable=sums complete

                    for (int ich = 0; ich < CONV2_IN_C; ich++) {
                        ap_uint<32> input_packed = 0;
                        for (int k = 0; k < 9; k++) {
                            #pragma HLS UNROLL
                            input_packed[k] = window[ich][k/3][k%3];
                        }

                        for (int och_off = 0; och_off < 2; och_off++) {
                            #pragma HLS UNROLL
                            int och = och_base + och_off;
                            if (och < CONV2_OUT_C) {
                                ap_uint<32> weight_packed = 0;
                                for (int k = 0; k < 9; k++) {
                                    #pragma HLS UNROLL
                                    weight_packed[k] = conv2_weights[och][ich][k/3][k%3];
                                }
                                sums[och_off] += xnor_popcount9(input_packed, weight_packed);
                            }
                        }
                    }

                    for (int och_off = 0; och_off < 2; och_off++) {
                        int och = och_base + och_off;
                        if (och < CONV2_OUT_C) {
                            ap_int<16> result = sums[och_off];
                            if (result > 127) result = 127;
                            else if (result < -128) result = -128;

                            ap_int<8> out_val;
                            if      (result >  0) out_val =  1;
                            else if (result <  0) out_val = -1;
                            else                  out_val =  0;
                            output_data[out_idx + och] = out_val;
                        }
                    }
                }
                out_idx += CONV2_OUT_C;
            }
        }
    }
}

// --------------------------------------------------------
// 6) Conv3 Layer
// --------------------------------------------------------
void conv3_layer(
    ap_int<8> input_data[CONV2_OUT_H * CONV2_OUT_W * CONV2_OUT_C],
    ap_int<8> output_data[CONV3_OUT_H * CONV3_OUT_W * CONV3_OUT_C]
) {
    #pragma HLS INLINE off

    ap_uint<1> line_buffer[CONV3_IN_C][3][CONV2_OUT_W + 2];
    #pragma HLS ARRAY_PARTITION variable=line_buffer dim=1 cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=line_buffer dim=2 complete
    #pragma HLS RESOURCE variable=line_buffer core=RAM_1P_BRAM

    ap_uint<1> window[CONV3_IN_C][3][3];
    #pragma HLS ARRAY_PARTITION variable=window dim=1 cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=window dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=window dim=3 complete

    #pragma HLS RESOURCE variable=conv3_weights core=ROM_1P
    #pragma HLS ARRAY_PARTITION variable=conv3_weights dim=1 cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=conv3_weights dim=2 cyclic factor=4

    // 버퍼 초기화
    for (int ch = 0; ch < CONV3_IN_C; ch++) {
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < CONV2_OUT_W + 2; c++) {
                line_buffer[ch][r][c] = 0;
            }
        }
    }

    int out_idx = 0;
    for (int row = 0; row < CONV2_OUT_H; row++) {
        for (int col = 0; col < CONV2_OUT_W; col++) {
            for (int ch_base = 0; ch_base < CONV3_IN_C; ch_base += 2) {
                #pragma HLS PIPELINE II=1

                for (int ch_off = 0; ch_off < 2; ch_off++) {
                    #pragma HLS UNROLL
                    int ch = ch_base + ch_off;
                    if (ch < CONV3_IN_C) {
                        ap_int<8> pixel = input_data[(row * CONV2_OUT_W + col) * CONV3_IN_C + ch];
                        ap_uint<1> bit = (pixel >= 0) ? 1 : 0;

                        line_buffer[ch][2][col + 1] = line_buffer[ch][1][col + 1];
                        line_buffer[ch][1][col + 1] = line_buffer[ch][0][col + 1];
                        line_buffer[ch][0][col + 1] = bit;

                        for (int r = 0; r < 3; r++) {
                            #pragma HLS UNROLL
                            window[ch][r][0] = window[ch][r][1];
                            window[ch][r][1] = window[ch][r][2];
                            window[ch][r][2] = line_buffer[ch][r][col + 1];
                        }
                    }
                }
            }

            if (row >= 2 && col >= 2) {
                for (int och = 0; och < CONV3_OUT_C; och++) {
                    #pragma HLS PIPELINE II=1

                    ap_int<16> sum = 0;

                    for (int ich_base = 0; ich_base < CONV3_IN_C; ich_base += 2) {
                        for (int ich_off = 0; ich_off < 2; ich_off++) {
                            int ich = ich_base + ich_off;
                            if (ich < CONV3_IN_C) {
                                ap_uint<32> input_packed = 0;
                                ap_uint<32> weight_packed = 0;

                                for (int k = 0; k < 9; k++) {
                                    #pragma HLS UNROLL
                                    input_packed[k] = window[ich][k/3][k%3];
                                    weight_packed[k] = conv3_weights[och][ich][k/3][k%3];
                                }

                                sum += xnor_popcount9(input_packed, weight_packed);
                            }
                        }
                    }

                    if (sum > 127) sum = 127;
                    else if (sum < -128) sum = -128;

                    ap_int<8> out_val;
                    if      (sum >  0) out_val =  1;
                    else if (sum <  0) out_val = -1;
                    else               out_val =  0;
                    output_data[out_idx + och] = out_val;
                }
                out_idx += CONV3_OUT_C;
            }
        }
    }
}

// --------------------------------------------------------
// 7) Pooling Layer
// --------------------------------------------------------
void pooling_layer(
    ap_int<8> input_data[CONV3_OUT_H * CONV3_OUT_W * CONV3_OUT_C],
    ap_int<8> output_data[POOL_OUT_H * POOL_OUT_W * CONV3_OUT_C]
) {
    #pragma HLS INLINE off
    #pragma HLS RESOURCE variable=input_data core=RAM_2P_BRAM
    #pragma HLS RESOURCE variable=output_data core=RAM_2P_BRAM

    int out_idx = 0;
    for (int row = 0; row < CONV3_OUT_H; row += POOL_K) {
        for (int col = 0; col < CONV3_OUT_W; col += POOL_K) {
            for (int ch = 0; ch < CONV3_OUT_C; ch++) {
                ap_int<8> w0 = input_data[((row + 0) * CONV3_OUT_W + (col + 0)) * CONV3_OUT_C + ch];
                ap_int<8> w1 = input_data[((row + 0) * CONV3_OUT_W + (col + 1)) * CONV3_OUT_C + ch];
                ap_int<8> w2 = input_data[((row + 1) * CONV3_OUT_W + (col + 0)) * CONV3_OUT_C + ch];
                ap_int<8> w3 = input_data[((row + 1) * CONV3_OUT_W + (col + 1)) * CONV3_OUT_C + ch];

                ap_int<10> sum = w0 + w1 + w2 + w3;
                ap_int<8> pooled = sum >> 2;
                output_data[out_idx + ch] = pooled;
            }
            out_idx += CONV3_OUT_C;
        }
    }
}

// --------------------------------------------------------
// 8) FC Layer
// --------------------------------------------------------
void fc_layer(
    ap_int<8> input_data[FC_INPUT],
    FC_TYPE output_data[FC_OUTPUT]
) {
    #pragma HLS INLINE off
    #pragma HLS RESOURCE variable=fc_weights core=ROM_1P
    #pragma HLS RESOURCE variable=fc_bias core=ROM_1P
    #pragma HLS ARRAY_PARTITION variable=input_data cyclic factor=4

    ap_int<16> fc_results[FC_OUTPUT];
    #pragma HLS ARRAY_PARTITION variable=fc_results complete

    // bias 초기화
    for (int i = 0; i < FC_OUTPUT; i++) {
        #pragma HLS UNROLL factor=5
        fc_results[i] = fc_bias[i];
    }

    // XNOR 연산
    for (int i = 0; i < FC_INPUT; i += 32) {
        #pragma HLS UNROLL factor=11
        ap_uint<32> input_chunk = 0;

        for (int j = 0; j < 32 && (i + j) < FC_INPUT; j++) {
            ap_uint<1> bit = (input_data[i + j] >= 0) ? 1 : 0;
            input_chunk[j] = bit;
        }

        for (int out = 0; out < FC_OUTPUT; out++) {
            ap_uint<32> weight_chunk = 0;
            for (int j = 0; j < 32 && (i + j) < FC_INPUT; j++) {
                weight_chunk[j] = fc_weights[out][i + j];
            }
            fc_results[out] += xnor_popcount(input_chunk, weight_chunk);
        }
    }

    // 출력
    for (int i = 0; i < FC_OUTPUT; i++) {
        #pragma HLS UNROLL factor=5
        output_data[i] = (FC_TYPE)fc_results[i];
    }
}

// --------------------------------------------------------
// 9) 단일 샘플 처리 함수
// --------------------------------------------------------
void process_single_sample(
    D_TYPE input_buffer[25],
    FC_TYPE fc_output[FC_OUTPUT],
    ap_uint<1> binary_pixels[IMG_H*IMG_W],
    ap_int<8> conv1_output[CONV1_OUT_H*CONV1_OUT_W*CONV1_OUT_C],
    ap_int<8> conv2_output[CONV2_OUT_H*CONV2_OUT_W*CONV2_OUT_C],
    ap_int<8> conv3_output[CONV3_OUT_H*CONV3_OUT_W*CONV3_OUT_C],
    ap_int<8> pool_output[POOL_OUT_H*POOL_OUT_W*CONV3_OUT_C]
) {
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW

    unpack_binary_input(input_buffer, binary_pixels);
    conv1_layer(binary_pixels, conv1_output);
    conv2_layer(conv1_output, conv2_output);
    conv3_layer(conv2_output, conv3_output);
    pooling_layer(conv3_output, pool_output);
    fc_layer(pool_output, fc_output);
}

// ==================== Top function ====================
void mnist_cnn_top(
    AXI_VAL input_stream[BATCH_SIZE * 25],
    AXI_VAL output_stream[BATCH_SIZE * FC_OUTPUT]
) {
    #pragma HLS INTERFACE s_axilite port=return
    #pragma HLS INTERFACE axis port=input_stream
    #pragma HLS INTERFACE axis port=output_stream

    // 전체 입력 및 출력 버퍼를 static으로 선언
    static D_TYPE input_buffer[BATCH_SIZE * 25];
    #pragma HLS RESOURCE variable=input_buffer core=RAM_2P_BRAM

    static FC_TYPE fc_output[BATCH_SIZE * FC_OUTPUT];
    #pragma HLS RESOURCE variable=fc_output core=RAM_2P_BRAM

    // 입력 데이터 로드
    load_input(input_buffer, input_stream);

    // 각 배치 샘플을 독립적으로 처리
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        // 현재 배치용 로컬 버퍼들 선언
        D_TYPE current_input[25];
        #pragma HLS ARRAY_PARTITION variable=current_input complete

        FC_TYPE current_output[FC_OUTPUT];
        #pragma HLS ARRAY_PARTITION variable=current_output complete

        // 현재 배치의 입력 추출
        for (int i = 0; i < 25; i++) {
            #pragma HLS UNROLL
            current_input[i] = input_buffer[batch * 25 + i];
        }

        // 중간 버퍼들 - static으로 선언하여 재사용
        static ap_uint<1> binary_pixels[IMG_H*IMG_W];
        static ap_int<8> conv1_output[CONV1_OUT_H*CONV1_OUT_W*CONV1_OUT_C];
        static ap_int<8> conv2_output[CONV2_OUT_H*CONV2_OUT_W*CONV2_OUT_C];
        static ap_int<8> conv3_output[CONV3_OUT_H*CONV3_OUT_W*CONV3_OUT_C];
        static ap_int<8> pool_output[POOL_OUT_H*POOL_OUT_W*CONV3_OUT_C];

        #pragma HLS RESOURCE variable=binary_pixels core=RAM_1P_BRAM
        #pragma HLS RESOURCE variable=conv1_output core=RAM_2P_BRAM
        #pragma HLS RESOURCE variable=conv2_output core=RAM_2P_BRAM
        #pragma HLS RESOURCE variable=conv3_output core=RAM_2P_BRAM
        #pragma HLS RESOURCE variable=pool_output core=RAM_2P_BRAM

        // 각 버퍼 초기화 (중요!)
        for (int i = 0; i < IMG_H*IMG_W; i++) {
            #pragma HLS PIPELINE
            binary_pixels[i] = 0;
        }

        // 단일 샘플 처리
        process_single_sample(
            current_input,
            current_output,
            binary_pixels,
            conv1_output,
            conv2_output,
            conv3_output,
            pool_output
        );

        // 결과 저장
        for (int i = 0; i < FC_OUTPUT; i++) {
            #pragma HLS UNROLL
            fc_output[batch * FC_OUTPUT + i] = current_output[i];
        }
    }

    // 출력 스트림으로 전송
    store_output(output_stream, fc_output);
}
