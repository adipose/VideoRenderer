inline float3 hable(float3 x)
{
    const float A = 0.15, B = 0.50, C = 0.10, D = 0.20, E = 0.02, F = 0.30;

    return ((x * (A * x + (C * B)) + (D * E)) / (x * (A * x + B) + (D * F))) - E / F;
}

float3 ToneMappingHable(const float3 rgb)
{
    static const float3 HABLE_DIV = hable(4.8);

    return hable(rgb) / HABLE_DIV;
}

// ---- Spline tone mapping ------------------------------------------------------------------------
// The same curve and color model the HDR10 local tone mapping uses, in a form the HDR to SDR
// conversion can call: BT.2020 linear light, nits in and nits out, content peak P onto display
// peak D.  Hable above knows nothing about how bright the content is, so it compresses a 400 nit
// scene as hard as a 4000 nit one; this does not.
//
//   the gray curve  the Hermite spline of BT.2390 in PQ, knee at 1.5 * PQ(D) - 0.5 so it follows
//                   the display and not the content, flat at max(P, 4000 nits)
//   color           the curve is given a saturation weighted mix of luminance and largest channel,
//                   mixed towards a per channel curve in LMS so very bright colors drift in hue
//                   and turn paler instead of clipping
static const float kSplineFlatAt = 4000.0f;
static const float kSplineLmsShare = 0.17f;
static const float kSplineFadeGain = 0.415f;
static const float kSplineFadePower = 1.29f;
static const float3x3 kSplineRgbToLms = float3x3(
	0.412109375f, 0.523925781f, 0.063964844f,
	0.166748047f, 0.720458984f, 0.112792969f,
	0.024169922f, 0.075439453f, 0.900390625f);
static const float3x3 kSplineLmsToRgb = float3x3(
	 3.436606694f, -2.506452119f,  0.069845424f,
	-0.791329556f,  1.983600452f, -0.192270896f,
	-0.025949900f, -0.098913715f,  1.124863614f);

inline float SplineCurve(float nits, float knee, float width, float endVal, float top)
{
	const float e = LinearToST2084(nits, 10000.0f).x;
	if (e <= knee)
		return nits;
	const float t = saturate((e - knee) / width);
	const float t2 = t * t;
	const float t3 = t2 * t;
	const float h = (2.0f * t3 - 3.0f * t2 + 1.0f) * knee + (t3 - 2.0f * t2 + t) * width + (-2.0f * t3 + 3.0f * t2) * endVal;
	return ST2084ToLinear(min(h, top), 10000.0f).x;
}

float3 ToneMappingSpline(float3 nits, float D, float P)
{
	const float M = max(nits.r, max(nits.g, nits.b));
	if (M <= 0.000001f || P <= D)
		return nits;

	const float top = LinearToST2084(D, 10000.0f).x;
	const float knee = max(1.5f * top - 0.5f, 0.0f);
	const float width = LinearToST2084(max(P, kSplineFlatAt), 10000.0f).x - knee;
	float endVal = top;
	if (P < kSplineFlatAt) {
		// the value at the content peak is linear in the end value, so it can be solved for
		const float t = (LinearToST2084(P, 10000.0f).x - knee) / width;
		const float t2 = t * t;
		const float t3 = t2 * t;
		endVal = (top - (2.0f * t3 - 3.0f * t2 + 1.0f) * knee - (t3 - 2.0f * t2 + t) * width) / (-2.0f * t3 + 3.0f * t2);
	}

	const float Y = 0.2627f * nits.r + 0.6780f * nits.g + 0.0593f * nits.b;
	if (Y <= 0.000001f)
		return nits;
	const float s = saturate(1.0f - min(nits.r, min(nits.g, nits.b)) / M);
	const float s2 = s * s;
	const float N = Y * pow(max(M / Y, 1.0f), s2 * s2);
	const float3 S = nits * (SplineCurve(N, knee, width, endVal, top) / N);

	const float3 lms = max(mul(kSplineRgbToLms, nits), 0.0f);
	const float3 PC = max(mul(kSplineLmsToRgb, float3(
		SplineCurve(lms.x, knee, width, endVal, top),
		SplineCurve(lms.y, knee, width, endVal, top),
		SplineCurve(lms.z, knee, width, endVal, top))), 0.0f);

	const float rho = max(max(S.r, max(S.g, S.b)) / D, 1.0f);
	const float mu = kSplineLmsShare + (1.0f - kSplineLmsShare) * (1.0f - 1.0f / (rho * rho));
	float3 c = max(lerp(S, PC, mu), 0.0f);

	const float compress = saturate(1.0f - D / P);
	const float Yout = 0.2627f * c.r + 0.6780f * c.g + 0.0593f * c.b;
	const float k = min(kSplineFadeGain * compress * pow(max((c.r + c.g + c.b) / (3.0f * D), 0.0f), kSplineFadePower), 1.0f);
	c = Yout + (c - Yout) * (1.0f - k);

	const float m = max(c.r, max(c.g, c.b));
	return (m > D) ? c * (D / m) : c;
}

// rgb is scaled so that 1.0 is the display's white; param2 carries the content peak in nits, and
// is 0 when the option is off, which keeps the old fixed curve.
float3 ToneMappingSdr(float3 rgb, float luminanceScale, float contentNits)
{
	if (contentNits <= 0.0f)
		return ToneMappingHable(rgb);
	const float D = 10000.0f / max(luminanceScale, 0.0001f);
	return ToneMappingSpline(rgb * D, D, contentNits) / D;
}
