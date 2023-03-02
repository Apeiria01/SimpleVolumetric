#include "VolumetricData.h"
unsigned char* loadRawFile(const char* filename, size_t size) {
	FILE* fp = fopen(filename, "rb");

	if (!fp) {
		fprintf(stderr, "Error opening file '%s'\n", filename);
		return 0;
	}

	auto data = new unsigned char[size];
	size_t read = fread(data, 1, size, fp);
	fclose(fp);

#if defined(_MSC_VER_)
	printf("Read '%s', %Iu bytes\n", filename, read);
#else
	printf("Read '%s', %zu bytes\n", filename, read);
#endif

	return data;
}