#pragma once

#include <vector>

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
	* Destructor
	*/
	~GPUArray();

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
	void download(unsigned char* memPointer) const;

	/*
	* Changes the dimensions of the array
	*
	* @param arrayShape: Dimensions of the array (e.g. {2, 3, 4} for a 3D array with 2 layers, 3 rows and 4 columns)
	* @param initializer: Value to initialize all array entries with
	* @param bytesRequest: Size of the array in bytes (optional)
	*/
	void changeDims(std::vector<unsigned int> arrayShape, T initializer = 0, size_t bytesRequest = 0);

	/*
	* Initializes the array with the provided dimensions
	*
	* @param arrayShape: Dimensions of the array (e.g. {2, 3, 4} for a 3D array with 2 layers, 3 rows and 4 columns)
	* @param initializer: Value to initialize all array entries with
	* @param bytesRequest: Size of the array in bytes (optional)
	*/
	void init(std::vector<unsigned int> arrayShape, T initializer = 0, size_t bytesRequest = 0) {
		changeDims(arrayShape, initializer, bytesRequest);
	}

	/*
	* Returns whether the array is initialized (i.e. has allocated memory)
	*
	* @return: True if the array is initialized, false otherwise
	*/
	bool isInitialized() const;

	/*
	* Sets every array entry to 0
	*/
	void zero() const;

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
	void fill(T value, unsigned int startIdx, unsigned int endIndex);

	/*
	* Fills the array with data from system memory
	*
	* @param memPointer: Pointer to the data in system memory
	*/
	void fillData(const unsigned char* memPointer) const;

	/*
	* Prints a 2D array/matrix
	*
	* @param startIdx: Index of the first array entry to print
	* @param numElements: How many array entries to print (-1 : all will be printed)
	*/
	template <typename S>
	void print(unsigned int startIdx, int numElements);
};