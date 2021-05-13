#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <math.h>
using namespace hls;
typedef ap_axiu<32, 0, 0, 0> AP_AXIS;
typedef stream<AP_AXIS> AXI_STREAM;

void graph(AXI_STREAM &in_strm, AXI_STREAM &out_strm) {
	float xarg0[4];
	#pragma HLS INTERFACE s_axilite port=return bundle=CONTROL_BUS
	#pragma HLS INTERFACE axis port=in_strm
	#pragma HLS INTERFACE axis port=out_strm
	union
	{
	    unsigned int as_uint32;
	    float as_floatingpoint;
	} my_converter;
	AP_AXIS element;

	for (int i = 0; i < 4; i++) {
		element = in_strm.read();
		my_converter.as_uint32 = element.data;
		xarg0[i] = my_converter.as_floatingpoint;
	}
	float xcst = 0.0;
	float xcst_0 = -INFINITY;
	long xc0_i64 = 0;
	long xc4_i64 = 4;
	long xc8_i64 = 8;
	long xc20_i64 = 20;
	long xc32_i64 = 32;
	long xc72_i64 = 72;
	long xc112_i64 = 112;
	char x0[152];
	float x1;
	float x2;
	float x3[3];
	float x4[3];
	float x5[3];
	float x6[10];
	float x7[10];
	float x8[10];
	float x9[10];
	float x10[10];
	float x11[10];
	float x12[4][10] = {{-0.24613583,-0.092899084,0.6192079,-0.4243827,-0.4434815,-0.9284553,1.2941761,0.3020994,-0.4197688,0.26942998},{0.13739973,0.22751433,0.62437403,0.13095987,0.18704581,-0.7885614,1.1330024,-0.6823718,-0.33963132,-0.40294567},{-0.023765981,-0.4105407,-0.54323965,-0.133308,0.051795542,2.311332,-0.52598023,0.350284,-0.3883608,-0.43208462},{-0.20888734,-0.24245977,-0.79783046,-0.07657069,0.17619997,2.1291342,-1.0830684,0.32420763,0.047249377,-0.3567633}};
	for (int xarg1 = 0; xarg1 < 10; xarg1++) {
		float x20;
		x20 = xcst;
		for (int xarg2 = 0; xarg2 < 4; xarg2++) {
			float x22 = xarg0[xarg2];
			float x23 = x12[xarg2][xarg1];
			float x24 = x20;
			float x25 = x22 * x23;
			float x26 = x24 + x25;
			x20 = x26;
		}
		float x21 = x20;
		x11[xarg1] = x21;
	}
	float x13[10] = {-0.31164688,-0.4312213,0.4697529,0.48109454,0.27246112,-0.24887842,0.2797409,-0.33201337,0.187311,0.38295975};
	for (int xarg1 = 0; xarg1 < 10; xarg1++) {
		float x20 = x11[xarg1];
		float x21 = x13[xarg1];
		float x22 = x20 + x21;
		x10[xarg1] = x22;
	}
	for (int xarg1 = 0; xarg1 < 10; xarg1++) {
		float x20 = x10[xarg1];
		bool x21 = x20 < xcst;
		float x22 = x21 ? xcst : x20;
		x9[xarg1] = x22;
	}
	float x14[10][10] = {{-0.053294927,0.22808722,0.12433183,0.2747425,0.24589697,-0.10145347,-0.04131329,0.018805057,-0.29297155,0.17166856},{-0.03848943,0.26574245,-0.18430483,0.16724506,-0.13140449,0.28065088,0.09902784,-0.07555905,0.1581473,0.14666668},{0.0294935,0.122816294,0.11018929,0.029992566,0.9365798,-0.22937669,-0.090096325,0.21203026,-0.08861856,0.16418213},{0.21572515,0.05591333,-0.29715154,0.19740888,-0.0643003,-0.22864464,-0.073579594,0.08896476,-0.06188658,0.2511004},{0.30568424,-0.14220431,-0.07088666,0.029296905,-0.08507678,-0.12107113,-0.21168256,-0.020436347,-0.17885904,0.2262449},{0.292502,-0.1281471,0.010785878,-0.00053620443,-1.7213602,1.6306708,-0.23303139,-0.038422287,0.08013456,-0.2105258},{0.0655615,-0.1562882,-0.15713876,0.011560181,1.6676966,0.1366701,-0.2490134,-0.12197373,-0.14636132,-0.05800635},{-0.0014007185,0.07811475,-0.20760767,0.029896764,-0.30818552,0.48145798,0.11561197,0.25193498,0.075928316,0.0022871662},{-0.21086758,0.10870299,0.30000302,-0.29136324,0.063106,0.14354098,0.110801876,0.26896605,-0.0713066,-0.31040135},{0.27607504,-0.27579218,0.19764844,0.04605058,0.07150844,0.042691763,0.10005078,0.05559138,-0.2080051,-0.27086735}};
	for (int xarg1 = 0; xarg1 < 10; xarg1++) {
		float x20;
		x20 = xcst;
		for (int xarg2 = 0; xarg2 < 10; xarg2++) {
			float x22 = x9[xarg2];
			float x23 = x14[xarg2][xarg1];
			float x24 = x20;
			float x25 = x22 * x23;
			float x26 = x24 + x25;
			x20 = x26;
		}
		float x21 = x20;
		x8[xarg1] = x21;
	}
	float x15[10] = {-0.17639515,-0.106703,-0.23613828,-0.29251605,1.0263213,-0.2573433,0.14860874,0.044274867,0.040618856,0.05377919};
	for (int xarg1 = 0; xarg1 < 10; xarg1++) {
		float x20 = x8[xarg1];
		float x21 = x15[xarg1];
		float x22 = x20 + x21;
		x7[xarg1] = x22;
	}
	for (int xarg1 = 0; xarg1 < 10; xarg1++) {
		float x20 = x7[xarg1];
		bool x21 = x20 < xcst;
		float x22 = x21 ? xcst : x20;
		x6[xarg1] = x22;
	}
	float x16[10][3] = {{-0.33576977,-0.029962858,0.0690972},{-0.27757403,0.0014446378,-0.0067016482},{-0.006242752,0.2870156,-0.09103586},{-0.062724926,0.2808149,0.05601872},{1.320392,0.97525275,-1.8274055},{-1.5312539,0.35198757,1.0384022},{0.15396473,-0.20561242,-0.1378246},{0.1472234,0.11049871,0.2509049},{0.24624053,-0.01159292,-0.16513237},{-0.23420364,-0.2446965,-0.11065629}};
	for (int xarg1 = 0; xarg1 < 3; xarg1++) {
		float x20;
		x20 = xcst;
		for (int xarg2 = 0; xarg2 < 10; xarg2++) {
			float x22 = x6[xarg2];
			float x23 = x16[xarg2][xarg1];
			float x24 = x20;
			float x25 = x22 * x23;
			float x26 = x24 + x25;
			x20 = x26;
		}
		float x21 = x20;
		x5[xarg1] = x21;
	}
	float x17[3] = {0.2560478,0.03993625,-0.4343873};
	for (int xarg1 = 0; xarg1 < 3; xarg1++) {
		float x20 = x5[xarg1];
		float x21 = x17[xarg1];
		float x22 = x20 + x21;
		x4[xarg1] = x22;
	}
	x2 = xcst;
	x1 = xcst_0;
	for (int xarg1 = 0; xarg1 < 3; xarg1++) {
		float x20 = x1;
		float x21 = x4[xarg1];
		bool x22 = x20 > x21;
		float x23 = x22 ? x20 : x21;
		x1 = x23;
	}
	float x18 = x1;
	for (int xarg1 = 0; xarg1 < 3; xarg1++) {
		float x20 = x2;
		float x21 = x4[xarg1];
		float x22 = x21 - x18;
		float x23 = exp(x22);
		float x24 = x20 + x23;
		x2 = x24;
		x3[xarg1] = x23;
	}
	float x19 = x2;
	for (int xarg1 = 0; xarg1 < 3; xarg1++) {
		float x20 = x3[xarg1];
		float x21 = x20 / x19;
		x3[xarg1] = x21;
	}
	AP_AXIS val;
	val.keep = element.keep;
	val.strb = element.strb;
	val.last = 0;

	for (int i = 0; i < 3; i++) {
		my_converter.as_floatingpoint = x3[i];
		val.data = my_converter.as_uint32;
		if(i == 2) val.last = 1;
		out_strm << val;
	}
	return;
}
