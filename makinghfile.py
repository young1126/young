import numpy as np

def convert_npy_to_weights_only(npy_path, output_header_path):
    """
    가중치만 포함한 간단한 헤더 파일 생성
    """
    
    # NPY 파일 로드
    weights_dict = np.load(npy_path, allow_pickle=True).item()
    
    conv1_weights = weights_dict['conv1']
    conv2_weights = weights_dict['conv2']
    conv3_weights = weights_dict['conv3']
    fc_weights = weights_dict['fc_w']
    fc_bias = weights_dict['fc_b']
    fc_w_4d = fc_weights.reshape(10, 32, 11, 11)             # [out, C, H, W]
    fc_w_space_first = fc_w_4d.transpose(0, 2, 3, 1)       # [out, H, W, C]
    fc_w_space_first = fc_w_space_first.reshape(10, -1) 
    # 헤더 파일 생성
    with open(output_header_path, 'w') as f:
        f.write("#ifndef WEIGHTS_H\n")
        f.write("#define WEIGHTS_H\n\n")
        f.write("#include \"ap_int.h\"\n\n")
        
        # Conv1 weights [16][1][3][3]
        f.write("static const ap_int<8> conv1_weights[16][1][3][3] = {\n")
        for out_ch in range(16):
            f.write("    { { ")
            for h in range(3):
                f.write("{ ")
                for w in range(3):
                    val = int(conv1_weights[out_ch, 0, h, w])
                    f.write(f"{val}")
                    if w < 2: f.write(", ")
                f.write(" }")
                if h < 2: f.write(", ")
            f.write(" } }")
            if out_ch < 15: f.write(",")
            f.write("\n")
        f.write("};\n\n")
        
        # Conv2 weights [16][16][3][3]
        f.write("static const ap_uint<1> conv2_weights[16][16][3][3] = {\n")
        for out_ch in range(16):
            f.write("    { ")
            for in_ch in range(16):
                f.write("{ ")
                for h in range(3):
                    f.write("{ ")
                    for w in range(3):
                        val = int(conv2_weights[out_ch, in_ch, h, w])
                        f.write(f"{val}")
                        if w < 2: f.write(", ")
                    f.write(" }")
                    if h < 2: f.write(", ")
                f.write(" }")
                if in_ch < 15: f.write(", ")
            f.write(" }")
            if out_ch < 15: f.write(",")
            f.write("\n")
        f.write("};\n\n")
        
        # Conv3 weights [32][16][3][3]
        f.write("static const ap_uint<1> conv3_weights[32][16][3][3] = {\n")
        for out_ch in range(32):
            f.write("    { ")
            for in_ch in range(16):
                f.write("{ ")
                for h in range(3):
                    f.write("{ ")
                    for w in range(3):
                        val = int(conv3_weights[out_ch, in_ch, h, w])
                        f.write(f"{val}")
                        if w < 2: f.write(", ")
                    f.write(" }")
                    if h < 2: f.write(", ")
                f.write(" }")
                if in_ch < 15: f.write(", ")
            f.write(" }")
            if out_ch < 31: f.write(",")
            f.write("\n")
        f.write("};\n\n")
        
        # FC weights [10][3872]
        f.write("static const ap_uint<1> fc_weights[10][3872] = {\n")
        for out_idx in range(10):
            f.write("    { ")
            for in_idx in range(3872):
                val = int(fc_w_space_first[out_idx, in_idx])
                f.write(f"{val}")
                if in_idx < 3871: f.write(", ")
            f.write(" }")
            if out_idx < 9: f.write(",")
            f.write("\n")
        f.write("};\n\n")
        
        # FC bias [10]
        f.write("static const ap_int<8> fc_bias[10] = {\n    ")
        for i in range(10):
            val = int(fc_bias[i])
            f.write(f"{val}")
            if i < 9: f.write(", ")
        f.write("\n};\n\n")
        
        f.write("#endif\n")
    
    print(f"✅ Weights header generated: {output_header_path}")

if __name__ == "__main__":
    npy_file = "/home/jinseopalang/young/hls_friendly_weights_fixed.npy"
    header_file = "/home/jinseopalang/young/weights_3.h"
    
    convert_npy_to_weights_only(npy_file, header_file)