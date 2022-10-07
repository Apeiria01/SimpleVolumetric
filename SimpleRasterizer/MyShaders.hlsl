

SamplerState g_sampler : register(s0);
Texture2D g_texture : register(t0);

struct TexturedVertexPSInput
{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD;
};

TexturedVertexPSInput VSMain(float4 position : POSITION, float2 texCoord : TEXCOORD)
{
	TexturedVertexPSInput result;
	result.position = position;
	result.texCoord = texCoord;
	return result;
}

float4 PSMain(TexturedVertexPSInput input) : SV_TARGET
{
	return g_texture.Sample(g_sampler, input.texCoord);
	
	//return input.position;
}
