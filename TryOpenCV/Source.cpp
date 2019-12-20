#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <cmath>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

Mat White_Noise(Mat img) 
{
	Mat res;
	res = img.clone();
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			if (!rand() % 256)
			{
				res.at<Vec3b>(i, j)[0] = 255;
				res.at<Vec3b>(i, j)[1] = 255;
				res.at<Vec3b>(i, j)[2] = 255;
			}
		}
	return res;
}

Mat SaltPepper_Noise(Mat& img, int min, int max)
{
	Mat noise(img.size(), img.type());
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			noise.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
	int random;
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			if (rand() % 100 >= 50)
				for (int ch = 0; ch < 3; ++ch) {
					random = rand() % 256;
					if (random < min) noise.at<Vec3b>(i, j)[ch] = min;
					if (random > max) noise.at<Vec3b>(i, j)[ch] = max;
				}
	img += noise;
	return img;
}

Mat GeometricMean(Mat img)
{
	Mat res;
	res = img.clone();
	int Radx = 2;	int Rady = 2;
	int mn = 2 * Radx * 2 * Rady;
	for (int x = 0; x < img.cols; x++)
		for (int y = 0; y < img.rows; y++)
		{
			double resR = 1;
			double resG = 1;
			double resB = 1;
			for (int i = -Rady; i < Rady; i++)
				for (int j = -Radx; j < Radx; j++)
				{
					int X = min(x + j, img.cols - 1);
					X = max(X, 0);
					int Y = min(y + i, img.rows - 1);
					Y = max(Y, 0);
					Vec3b bgr = img.at<Vec3b>(Y, X);
					if (bgr[2])
						resR *= bgr[2];
					if (bgr[1])
						resG *= bgr[1];
					if (bgr[0])
						resB *= bgr[0];
				}
			int a = pow(resB, 1.0 / (9 * Radx * Rady));
			res.at<Vec3b>(y, x)[0] = min(pow(resB, 1.0 / (mn)), 255.0);
			res.at<Vec3b>(y, x)[1] = min(pow(resG, 1.0 / (mn)), 255.0);
			res.at<Vec3b>(y, x)[2] = min(pow(resR, 1.0 / (mn)), 255.0);
		}
	return res;
}

Mat ArifmeticMean(Mat img)
{
	Mat res;
	res = img.clone();
	int Radx = 2;	int Rady = 2;
	int mn = 2 * Radx * 2 * Rady;
	for (int x = 0; x < img.cols; x++)
		for (int y = 0; y < img.rows; y++)
		{
			double resR = 1.0;
			double resG = 1.0;
			double resB = 1.0;
			for (int i = -Rady; i < Rady; i++)
				for (int j = -Radx; j < Radx; j++)
				{
					int X = min(x + j, img.cols - 1);
					X = max(X, 0);
					int Y = min(y + i, img.rows - 1);
					Y = max(Y, 0);
					Vec3b bgr = img.at<Vec3b>(Y, X);
					resR += bgr[2];
					resG += bgr[1];
					resB += bgr[0];
				}
			res.at<Vec3b>(y, x)[0] = min((int)(resB / (mn)), 255);
			res.at<Vec3b>(y, x)[1] = min((int)(resG / (mn)), 255);
			res.at<Vec3b>(y, x)[2] = min((int)(resR / (mn)), 255);
		}
	return res;
}

Mat HarmonicMean(Mat img)
{
	Mat res;
	res = img.clone();
	int Radx = 2;
	int Rady = 2;
	float mn = (float)(2 * Radx * 2 * Rady);
	for (int x = 0; x < img.cols; x++)
		for (int y = 0; y < img.rows; y++)
		{
			double resR = 0.0;
			double resG = 0.0;
			double resB = 0.0;
			for (int i = -Rady; i < Rady; i++)
				for (int j = -Radx; j < Radx; j++)
				{
					int X = min(x + j, img.cols - 1);
					X = max(X, 0);
					int Y = min(y + i, img.rows - 1);
					Y = max(Y, 0);
					Vec3b bgr = img.at<Vec3b>(Y, X);
					resR += 1.0 / bgr[2];
					resG += 1.0 / bgr[1];
					resB += 1.0 / bgr[0];
				}
			res.at<Vec3b>(y, x)[0] = min((int)(mn / resB), 255);
			res.at<Vec3b>(y, x)[1] = min((int)(mn / resG), 255);
			res.at<Vec3b>(y, x)[2] = min((int)(mn / resR), 255);
		}
	return res;
}

void insertionSort(Vec3b* window, int size) {
	int i, j;
	Vec3b temp;
	for (i = 0; i < size * size; i++) {
		temp = window[i];
		for (j = i - 1; j >= 0 && (0.3 * temp.val[2] + 0.59 * temp.val[1] +
			0.11 * temp.val[0]) < (0.3 * window[j].val[2] +
				0.59 * window[j].val[1] +
				0.11 * window[j].val[0]);
			j--) {
			window[j + 1] = window[j];
		}
		window[j + 1] = temp;
	}
}
Mat MidpointFilter(Mat& img, int size) {
	Mat res;
	res = img.clone();
	int n = size * size;
	Vec3b* window = new Vec3b[n];

	int i, j;
	int height = img.rows;
	int width = img.cols;

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			int x = -1;
			for (int p = 0; p < size; p++) {
				for (int q = 0; q < size; q++) {
					int tmp1 = i + p;
					while (tmp1 >= height) tmp1--;
					int tmp2 = j + q;
					while (tmp2 >= width) tmp2--;
					window[++x] = img.at<Vec3b>(tmp1, tmp2);
					}
			}
			insertionSort(window, size);
			Vec3b min = window[0];
			Vec3b max = window[size - 1];
			Vec3b avg = (min + max) / 2;
			res.at<Vec3b>(i, j) = avg;
		}
	}
	delete[] window;
	return res;
}

float GetIntensity(Mat src)
{
	float res = 0;
	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
		{
			Vec3b bgr = src.at<Vec3b>(y, x);
			res += (bgr[0] + bgr[1] + bgr[2]);
		}
	res = res / (src.rows * src.cols);
	return res;
}

float GetContrast(Mat src)
{
	float res = 0;
	float M = GetIntensity(src);
	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
		{
			Vec3b bgr = src.at<Vec3b>(y, x);
			res += pow((bgr[0] + bgr[1] + bgr[2]) - M, 2);
		}
	res = sqrt(res / (src.rows * src.cols));
	return res;
}

float GetCov(Mat src1, Mat src2)
{
	float res = 0;
	float M1 = GetIntensity(src1);
	float M2 = GetIntensity(src2);
	for (int y = 0; y < src1.rows; y++)
		for (int x = 0; x < src1.cols; x++)
		{
			Vec3b bgr1 = src1.at<Vec3b>(y, x);
			Vec3b bgr2 = src2.at<Vec3b>(y, x);
			res += (bgr1[0] + bgr1[1] + bgr1[2] - M1) * (bgr2[0] + bgr2[1] + bgr2[2] - M2);
		}
	return res / (src1.rows * src1.cols);
}

float SSIM(Mat src1, Mat src2)
{
	return (2.f * GetIntensity(src1) * GetIntensity(src2)) * (2 * GetCov(src1, src2)) 
		/ ((GetIntensity(src1) * GetIntensity(src1) + GetIntensity(src2) * GetIntensity(src2)) * (GetContrast(src1) * GetContrast(src1) 
			+ GetContrast(src2) * GetContrast(src2)));
}

int main()
{
	srand(time(NULL));
	setlocale(LC_ALL, "Russian");
	Mat src;
	src = imread("myimage.jpg", IMREAD_UNCHANGED);
	if (!src.data)
	{
		cout << "Image not loaded";
		return -1;
	}
	imshow("Исходник", src);
	waitKey(500);

	Mat noiseimg = SaltPepper_Noise(src, 10, 240);
	imshow("Cоль и перец", noiseimg);

	//Mat noiseimg = White_Noise(src);
	//imshow("Белый шум", noiseimg);

	Mat geometricmeanimg = GeometricMean(noiseimg);
	imshow("Фильтр геометрического среднего", geometricmeanimg);
	float SSIM_geometricmean = SSIM(geometricmeanimg, src);
	cout << "SSIM-индекс для геометрического среднего: " << SSIM_geometricmean << endl;


	Mat arifmetimeanimg = ArifmeticMean(noiseimg);
	imshow("Фильтр арифметического среднего", arifmetimeanimg);
	float SSIM_arifmetimean = SSIM(arifmetimeanimg, src);
	cout << "SSIM-индекс для арифметического среднего: " << SSIM_arifmetimean << endl;


	Mat harmonicmeanimg = HarmonicMean(noiseimg);
	imshow("Фильтр гармонического среднего", harmonicmeanimg);
	float SSIM_harmonicmean = SSIM(harmonicmeanimg, src);
	cout << "SSIM-индекс для гармонического среднего:  " << SSIM_harmonicmean << endl;


	Mat	midpointimg = MidpointFilter(noiseimg, 2);
	imshow("Midpoint Filter", midpointimg);
	float SSIM_midpoint = SSIM(midpointimg, src);
	cout << "SSIM-индекс для midpoint filter " << SSIM_midpoint << endl;

	if (SSIM_geometricmean > SSIM_arifmetimean)
	{
		if (SSIM_geometricmean > SSIM_harmonicmean)
		{
			if (SSIM_geometricmean > SSIM_midpoint)			
				cout << "Лучший результат у фильтра геометрического среднего. " << endl;			
			else cout << "Лучший результат у midpoint filter. " << endl;
		}
		else
		{
			if (SSIM_harmonicmean > SSIM_midpoint)
				cout << "Лучший результат у фильтра гармонического среднего. " << endl;
			else cout << "Лучший результат у фильтра midpoint filter. " << endl;
		}
	}	else 
	{
		if (SSIM_arifmetimean > SSIM_harmonicmean)
		{
			if (SSIM_arifmetimean > SSIM_midpoint)
				cout << "Лучший результат у фильтра арифметического среднего. " << endl;
			else cout << "Лучший результат у фильтра midpoint filter. " << endl;
		}
		else
		{
			if (SSIM_harmonicmean > SSIM_midpoint)
				cout << "Лучший результат у фильтра гармонического среднего. " << endl;
			else cout << "Лучший результат у фильтра midpoint filter. " << endl;
		}
	}

	waitKey();
	return 0;
}

