// GPUArrayLib.cuh
#pragma once

// C++ libraries
#include <vector>

// Constants
constexpr int NUM_THREADS = 8;
const size_t SHMEM_SIZE = 3 * NUM_THREADS * NUM_THREADS;

/*
* This class initializes a multidimensional array on the GPU
*
* @param arrayShape: Dimensions of the array (e.g. {2, 3, 4} for a 3D array with 2 layers, 3 rows and 4 columns)
* @param initializer: Value to initialize all array entries with
*
* @return: GPUArray object
*
* @attr arrayPtrCPU: Pointer to the array on the CPU
* @attr arrayPtrGPU: Pointer to the array on the GPU
* @attr dims: Number of dimensions of the array
* @attr shape: Dimensions of the array
* @attr dimX: Dimension of the array in x direction
* @attr dimY: Dimension of the array in y direction
* @attr dimZ: Collected dimensions of the array beyond the first two dimensions
* @attr bytes: Size of the array in bytes
* @attr isOnGPU: Indicates if the array is currently on the GPU
*/
template <typename T>
class GPUArray {
public:
	T* arrayPtrCPU;
	T* arrayPtrGPU;
	unsigned int dims;
	std::vector<unsigned int> shape;
	unsigned int dimX;
	unsigned int dimY;
	unsigned int dimZ;
	size_t bytes;
	bool isOnGPU;

	/*
	* Default constructor for a standard multidimensional array
	*
	* @return: GPUArray object
	*/
	GPUArray();

	/*
	* Constructor for a standard multidimensional array
	*
	* @param arrayShape: Dimensions of the array (e.g. {2, 3, 4} for a 3D array with 2 layers, 3 rows and 4 columns)
	* @param initializer: Value to initialize all array entries with
	*
	* @return: GPUArray object
	*/
	GPUArray(std::vector<unsigned int> arrayShape, T initializer);

	/*
	* Constructor for an image array on the GPU
	*
	* @param filename: Path to the image file
	* @param height: Height of the image
	* @param width: Width of the image
	*
	*
	* @return: GPUArray object
	*/
	GPUArray(const char* filePath, unsigned int height, unsigned int width);

	/*
	* Constructor for an image batch array on the GPU
	*
	* @param filename: Path to the folder containing the image files
	* @param batchSize: Number of images in the batch
	* @param height: Height of the image
	* @param width: Width of the image	
	*
	* @return: GPUArray object
	*/
	GPUArray(const char* filePath, unsigned int batchSize, unsigned int height, unsigned int width);

	/*
	* Transfers the array to the GPU
	*/
	void toGPU();

	/*
	* Transfers the array to the CPU
	*/
	void toCPU();

	/*
	* Transfers the array to the provided memory pointer
	* 
	* @param memPointer: Pointer to the memory to transfer the array to
	*/
	void download(T* memPointer);

	/*
	* Changes the dimensions of the array
	*
	* @param arrayShape: Dimensions of the array (e.g. {2, 3, 4} for a 3D array with 2 layers, 3 rows and 4 columns)
	* @param initializer: Value to initialize all array entries with
	*/
	void changeDims(std::vector<unsigned int> arrayShape, T initializer = 0);

	/*
	* Initializes the array with the provided dimensions
	*
	* @param arrayShape: Dimensions of the array (e.g. {2, 3, 4} for a 3D array with 2 layers, 3 rows and 4 columns)
	*/
	void init(std::vector<unsigned int> arrayShape, T initializer = 0) {
		changeDims(arrayShape, initializer);
	}

	/*
	* Returns whether the array is initialized (i.e. has allocated memory)
	*
	* @return: True if the array is initialized, false otherwise
	*/
	bool isInitialized() const;

	/*
	* Copies the array and returns the copy
	*
	* @return: Copy of the array
	*/
	GPUArray<T> copy();

	/*
	* Adds a value to the array
	*
	* @param value: Value to add to the array
	*/
	void add(T value);

	/*
	* Subtracts a value from the array
	*
	* @param value: Value to subtract from the array
	*/
	void sub(T value);

	/*
	* Scales the array by a scalar
	*
	* @param scalar: Scalar to scale the array by
	*/
	void mul(T scalar);

	/*
	* Divides the array by a value
	*
	* @param value: Value to divide the array by
	*/
	void div(T value);

	/*
	* Adds another array to the array
	*
	* @param array: Array to add to the array
	*/
	void add(GPUArray<T>& array);

	/*
	* Subtracts another array from the array
	*
	* @param array: Array to subtract from the array
	*/
	void sub(GPUArray<T>& array);

	/*
	* Multiplies another array to the array (this is not matrix multiplication)
	*
	* @param array: Array to multiply with the array
	*/
	void elementMul(GPUArray<T>& array);

	/*
	* Sets every array entry to the provided value
	*
	* @param value: Value to set the array entries to
	*/
	void fill(T value);

	/*
	* Sets every array entry in the provided range to the provided value
	*
	* @param value: Value to set the array entries to
	* @param startIdx: Index of the first array entry to set
	* @param endIndex: Index of the last array entry to set
	*/
	void fill(T value, int startIdx, int endIndex);

	/*
	* Fills the array with data from system memory
	*
	* @param memPointer: Pointer to the data in system memory
	*/
	void fillData(const T* memPointer);

	/*
	* Prints a 2D array/matrix
	*
	* @param startIdx: Index of the first array entry to print
	* @param numElements: How many array entries to print (-1 : all will be printed)
	*/
	template <typename S>
	void print(const unsigned int startIdx, const int numElements);

	/*
	* Returns the absolute index of an array entry given the indices for each dimension
	*/
	int getAbsIndex(const std::initializer_list<int> indicesForEachDim) const;

	/*
	* Exports the array as a PNG image
	*
	* @param filePath: Path to where the image should be saved
	*/
	void exportPNG(const char* filePath);

	/*
	* Exports the array as a flow image
	*
	* @param filePath: Path to where the flow image should be saved
	* @param direction: Direction of the flow (0: x, 1: y)
	*/
	void exportFlowImage(const char* filePath, int direction);

	/*
	* Destructor
	*/
	void del() const;
};

/*
* -------------------- MATRIX - VALUE OPERATIONS --------------------
*/



/*
* Adds a value to an array
*
* @param array: Array to be added to
* @param value: Value to add to the array
*
* @return: Result array
*/
template <typename T>
GPUArray<T> add(GPUArray<T>& array, T value);

/*
* Subtracts a value from an array
*
* @param array: Array to be subtracted from
* @param value: Value to subtract from the array
*
* @return: Result array
*/
template <typename T>
GPUArray<T> sub(GPUArray<T>& array, T value);

/*
* Scales an array by a scalar
*
* @param array: Array to be scaled
* @param value: Scalar to scale the array by
*
* @return: Result array
*/
template <typename T>
GPUArray<T> mul(GPUArray<T>& array, T scalar);

/*
* Divides an array by a divisor value
*
* @param array: Array to be divided
* @param b: Divisor to divide the array by
*
* @return: Result array
*/
template <typename T>
GPUArray<T> div(GPUArray<T>& array, T value);



/*
* -------------------- MATRIX - MATRIX OPERATIONS --------------------
*/



/*
* Computes the sum of two arrays
*
* @param arrayA: First array
* @param arrayB: Second array
*
* @return: Result array
*/
template <typename T>
GPUArray<T> add(GPUArray<T>& arrayA, GPUArray<T>& arrayB);

/*
* Subtracts two arrays
*
* @param arrayA: First array
* @param arrayB: Second array
*
* @return: Result array
*/
template <typename T>
GPUArray<T> sub(GPUArray<T>& arrayA, GPUArray<T>& arrayB);

/*
* Computes the matrix multiplication of two arrays
*
* @param arrayA: First array
* @param arrayB: Second array
*
* @return: Result array
*/
template <typename T>
GPUArray<T> mul(GPUArray<T>& arrayA, GPUArray<T>& arrayB);

/*
* Multiplies two arrays together (this is not matrix multiplication)
*
* @param arrayA: First array
* @param arrayB: Second array
*
* @return: Result array
*/
template <typename T>
GPUArray<T> elementMul(GPUArray<T>& arrayA, GPUArray<T>& arrayB);

/*
* Computes the array addition of two arrays at the provided indices
*
* @param arrayA: First array
* @param arrayB: Second array
* @param i: Indexing array
*
* @return: Result array
*/
template <typename T>
GPUArray<T> addIndexed(GPUArray<T>& arrayA, GPUArray<T>& arrayB, GPUArray<int>& i);

/*
* Tests the memory of the GPU
*
* @param bytes: Number of bytes to allocate
*/
void memTest(size_t bytes);