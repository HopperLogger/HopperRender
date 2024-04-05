#include <windows.h>
#include <tchar.h>
#include <streams.h>
#include "iez.h"
#include "HopperRender.h"
#include "opticalFlowCalc.cuh"
#include "opticalFlowCalcSDR.cuh"
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <cuda_runtime_api.h>

int main(int argc, char* argv[]) {
	// Parse command-line arguments
	if (argc < 15) {
        std::cerr << "Usage: " << argv[0] << " sourceVideoFilePath outputVideoFilePath targetFPS speed calcResDiv numIterations numSteps frameBlurKernel flowBlurKernel frameOutput startTimeMin startTimeSec endTimeMin endTimeSec showPreview" << std::endl;
        return 1;
    }
    std::string inputVideoPath = argv[1];
	std::string outputVideoPath = argv[2];
	double targetFPS;
	std::stringstream(argv[3]) >> targetFPS;
	double speed;
	std::stringstream(argv[4]) >> speed;
	float resolutionDivider = std::atof(argv[5]);
	int numIterations = std::atoi(argv[6]);
	int numSteps = std::atoi(argv[7]);
	int frameBlurKernelSize = std::atoi(argv[8]);
	int flowBlurKernelSize = std::atoi(argv[9]);
	FrameOutput frameOutput = static_cast<FrameOutput>(std::atoi(argv[10]));
	int startTimeMin = std::atoi(argv[11]);
	int startTimeSec = std::atoi(argv[12]);
	int endTimeMin = std::atoi(argv[13]);
	int endTimeSec = std::atoi(argv[14]);
	bool showPreview = (std::string(argv[15]) == "True");

	// Initialize other needed variables
	int quality = 100;
	int codec = cv::VideoWriter::fourcc('H', '2', '6', '4');
	double targetFT = 1000.0 / targetFPS;
	auto perfTotalStart = std::chrono::high_resolution_clock::now();
	auto perfStart = std::chrono::high_resolution_clock::now();
	double perfDuration = 0;
	int perfMinutes = 0;
	int perfSeconds = 0;
	unsigned int iIntFrameCounter = 0;
	float fScalar;
	double currentSourceTime;
	double currentIntTime;
	unsigned int numIntFrames;
	unsigned char progress;
	double remainingDuration;

    // Open the video file
    cv::VideoCapture inputVideo(inputVideoPath);
    if (!inputVideo.isOpened()) {
        printf("Error: Unable to open the video file.");
        return -1;
    }

	// Get the properties of the video
	const unsigned int dimX = inputVideo.get(cv::CAP_PROP_FRAME_WIDTH);
	const unsigned int dimY = inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT);
	const double fps = inputVideo.get(cv::CAP_PROP_FPS) * speed;
	const double sourceFT = 1000.0 / fps;
	const unsigned int totalNumFrames = inputVideo.get(cv::CAP_PROP_FRAME_COUNT);
	cv::Size frameSize(dimX, dimY);
	if (endTimeMin == 0 && endTimeSec == 0) {
		endTimeSec = totalNumFrames;
	}

	// Check if the start time is valid
	int startTime = (startTimeMin * 60 + startTimeSec) * inputVideo.get(cv::CAP_PROP_FPS);
	int endTime = (endTimeMin * 60 + endTimeSec) * inputVideo.get(cv::CAP_PROP_FPS);
	if (startTime < 0 || startTime >= totalNumFrames) {
		printf("Error: Invalid start time.");
		return -1;
	}
	inputVideo.set(cv::CAP_PROP_POS_FRAMES, startTime);

	// Initialize the output video file
    cv::VideoWriter outputVideo(outputVideoPath, codec, targetFPS, frameSize);
	outputVideo.set(cv::VIDEOWRITER_PROP_QUALITY, quality);
    if (!outputVideo.isOpened()) {
        printf("Error: Unable to create the output video file.");
        return -1;
    }

	// Initialize HopperRender
	TCHAR tszName[] = _T("HopperRender");
	LPUNKNOWN punk = NULL;
	HRESULT phr = S_OK;
	CHopperRender hopperRender = CHopperRender(tszName, punk, &phr);
	hopperRender.m_iFrameOutput = frameOutput;
	hopperRender.m_iNumIterations = numIterations;
	hopperRender.m_iFrameBlurKernelSize = frameBlurKernelSize;
	hopperRender.m_iFlowBlurKernelSize = flowBlurKernelSize;
	hopperRender.m_iNumSteps = numSteps;
	hopperRender.m_iDimX = dimX;
	hopperRender.m_iDimY = dimY;
	hopperRender.m_fResolutionDivider = resolutionDivider;
	hopperRender.m_bExportMode = true;
	hopperRender.m_pofcOpticalFlowCalc = new OpticalFlowCalcSDR(dimY, dimX, 1.0, resolutionDivider);

	// Initialize the frame buffers
    cv::Mat inputFrame;
	cv::Mat outputFrame = cv::Mat::zeros(frameSize, CV_8UC3);
    unsigned char* bgrArrayGPU;
	unsigned char* nv12ArrayGPU;
	cudaMalloc(reinterpret_cast<void**>(&bgrArrayGPU), dimY * dimX * 3);
	cudaMalloc(reinterpret_cast<void**>(&nv12ArrayGPU), static_cast<size_t>(dimY * dimX * 1.5));
    
    // Read frames from the video
    while (inputVideo.read(inputFrame) && (hopperRender.m_iFrameCounter + startTime) < endTime) {
		// Convert the BGR frame to NV12 format
		convertBGRtoNV12(inputFrame.data, bgrArrayGPU, inputFrame.data, nv12ArrayGPU, dimY, dimX);

		// Reset the variables for the new frame
		fScalar = 0.0f;
		currentSourceTime = hopperRender.m_iFrameCounter * sourceFT;
		currentIntTime = static_cast<double>(iIntFrameCounter) * targetFT;
		numIntFrames = static_cast<unsigned int>(((currentSourceTime + sourceFT) - currentIntTime) / targetFT);
		hopperRender.m_iNumIntFrames = numIntFrames;
		
		// Calculate the progress
		if (hopperRender.m_iFrameCounter % 10 == 0) {
			auto perfEnd = std::chrono::high_resolution_clock::now();
			perfDuration = std::chrono::duration_cast<std::chrono::milliseconds>(perfEnd - perfStart).count();
			perfDuration /= 10.0;
			remainingDuration = perfDuration * (endTime - startTime - hopperRender.m_iFrameCounter);
			perfMinutes = remainingDuration / 60000;
            perfSeconds = (static_cast<int>(remainingDuration) % 60000) / 1000;
			progress = (static_cast<double>(hopperRender.m_iFrameCounter) / static_cast<double>(endTime - startTime)) * 100.0;
			printf("Computing frame %d/%d (%d%%) - %.3f ms per frame - Estimated time remaining: %d min %d sec - Frame Diff: %d\n",
				hopperRender.m_iFrameCounter, endTime - startTime, progress, perfDuration, perfMinutes, perfSeconds, hopperRender.m_iCurrentSceneChange);
			perfStart = std::chrono::high_resolution_clock::now();
		}
		//printf("%d\n", progress);

		// Interpolate the frames
		for (unsigned int iIntFrameNum = 0; iIntFrameNum < numIntFrames; iIntFrameNum++) {
			hopperRender.InterpolateFrame(inputFrame.data, outputFrame.data, fScalar, iIntFrameNum);
			fScalar += 1.0f / static_cast<float>(numIntFrames);
			iIntFrameCounter++;

			// Convert the P010 frame to BGR format
			convertP010toBGR(hopperRender.m_pofcOpticalFlowCalc->m_outputFrame.arrayPtrGPU, outputFrame.data, bgrArrayGPU, dimY, dimX);

			// Write the frame to the output video
			outputVideo.write(outputFrame);

			// Display the frame
			if (showPreview) {
				cv::imshow("Interpolation Preview", outputFrame);
				if (cv::waitKey(1) == 'q') {
					return 2;
				}
			}
		}
        hopperRender.m_iFrameCounter++;
    }
    
    // Release the video objects
    inputVideo.release();
	outputVideo.release();
    
    // Destroy any OpenCV windows
    cv::destroyAllWindows();

	// Calculate the total duration
	auto perfTotalEnd = std::chrono::high_resolution_clock::now();
	auto perfTotalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(perfTotalEnd - perfTotalStart).count();
	printf("Interpolation took %d s in total.\n", perfTotalDuration / 1000);
    
    return 0;
}