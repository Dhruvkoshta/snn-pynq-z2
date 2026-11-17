#ifndef SNN_COMBINED_CONFIG_H
#define SNN_COMBINED_CONFIG_H

// ==========================================================================
// Auto-Generated Hardware Configuration for PYNQ-Z2 SNN Accelerator
// Includes: Data Augmentation, Direct Encoding (T=8), Distillation & QAT INT8
// ==========================================================================

#define SNN_TIMESTEPS        8
#define SNN_ENCODING_MODE    1  // 1 = Direct Analog Current, 0 = Poisson Rate
#define SNN_NUM_CLASSES      10

// Input Stage (Direct Analog Current Feed)
#define CONV0_NUM_BYTES   1728
#define CONV0_SCALE_Q24   0x00009E8E // Scale: 0.00241937
#define CONV0_SQNR_DB     41.13f

// ResBlock 1 Stage 1
#define RES1_CONV1_NUM_BYTES   36864
#define RES1_CONV1_SCALE_Q24   0x0000AC44 // Scale: 0.00262856
#define RES1_CONV1_SQNR_DB     40.43f

// ResBlock 1 Stage 2
#define RES1_CONV2_NUM_BYTES   36864
#define RES1_CONV2_SCALE_Q24   0x0000A471 // Scale: 0.00250915
#define RES1_CONV2_SQNR_DB     40.88f

// Transition Stage 1 (64 -> 128)
#define CONV1_NUM_BYTES   73728
#define CONV1_SCALE_Q24   0x0000C7BE // Scale: 0.00304783
#define CONV1_SQNR_DB     39.15f

// ResBlock 2 Stage 1
#define RES2_CONV1_NUM_BYTES   147456
#define RES2_CONV1_SCALE_Q24   0x0000AAF9 // Scale: 0.00260884
#define RES2_CONV1_SQNR_DB     40.55f

// ResBlock 2 Stage 2
#define RES2_CONV2_NUM_BYTES   147456
#define RES2_CONV2_SCALE_Q24   0x0000B6BC // Scale: 0.00278830
#define RES2_CONV2_SQNR_DB     39.94f

// Transition Stage 2 (128 -> 256)
#define CONV2_NUM_BYTES   294912
#define CONV2_SCALE_Q24   0x0000B7E7 // Scale: 0.00280612
#define CONV2_SQNR_DB     39.88f

// ResBlock 3 Stage 1
#define RES3_CONV1_NUM_BYTES   589824
#define RES3_CONV1_SCALE_Q24   0x0000C9DF // Scale: 0.00308033
#define RES3_CONV1_SQNR_DB     39.09f

// ResBlock 3 Stage 2
#define RES3_CONV2_NUM_BYTES   589824
#define RES3_CONV2_SCALE_Q24   0x0000CBD7 // Scale: 0.00311036
#define RES3_CONV2_SQNR_DB     39.00f

// Final Output Classifier
#define FC_NUM_BYTES   40960
#define FC_SCALE_Q24   0x0000A3AF // Scale: 0.00249761
#define FC_SQNR_DB     40.88f

#endif // SNN_COMBINED_CONFIG_H
