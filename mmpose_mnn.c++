#include <vector>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <algorithm>
#include <memory>
#include <ctime>
#define pi 3.1415926
// using namespace std;


void xyxy2xywh(std::vector<float> &box){
	box[2] = box[2] - box[0] + 1;
	box[3] = box[3] - box[1] + 1;
}


void box2cs(const std::vector<float> & box, std::vector<float> & center, std::vector<float> & scale, std::vector<float> image_size){
	float input_h = image_size[0];
	float input_w = image_size[1];
	float aspect_ratio = input_h / input_w;

	float x = box[0];
	float y = box[1];
	float w = box[2];
	float h = box[3];
	float cx = x + w * 0.5;
	float cy = y + h * 0.5;
	center = {cx, cy};

	if(w > aspect_ratio * h)
		h = w * 1.0 / aspect_ratio;
	else if(w < aspect_ratio * h)
		w = h * aspect_ratio;
	float cw = w / 200.0 * 1.25;
	float ch = h / 200.0 * 1.25;
	scale = {cw, ch};

}


auto rotate_point(std::vector<float> pt, float angle_rad){
	auto sn = sin(angle_rad);
	auto cs = cos(angle_rad);
	auto new_x = pt[0] * cs - pt[1] * sn;
	auto new_y = pt[0] * sn + pt[1] * cs;
	std::vector<float> tmp = {new_x, new_y};
	return tmp;
}


auto _get_3rd_point(std::vector<float> a, std::vector<float> b){

	std::vector<float> direction;
	for (int i = 0; i< a.size(); i++){
		direction.push_back(a[i] - b[i]);
	}

	// auto direction = a - b;
	std::vector<float> tmp {-direction[1], direction[0]};
	std::vector<float> third_pt;
	for (int i = 0; i < b.size(); i++){
		third_pt.push_back(b[i] + tmp[i]);
	}
	// std::vector<float> third_pt = b + std::vector<float>{-direction[1], direction[0]};

	return third_pt; 
}


cv::Mat get_affine_transform(std::vector<float> & center, std::vector<float> & scale, 
	float rotation, std::vector<float> & image_size, bool inv=false){
	
	// center: [156.06 822.86]
	// scales: [2.3342 3.1123]
	// rotations: 0.0
	// image_size: [192, 256]

 	

	std::vector<float> shift = {0.0, 0.0};
	std::vector<float> scale_tmp;
	for(int i = 0; i < scale.size(); i++){
		scale_tmp.push_back(scale[i] * 200.0);
	}

	float src_w = scale_tmp[0];
	float dst_h = image_size[0];
	float dst_w = image_size[1];

	float rot_rad = pi * rotation / 180.0;
	std::vector<float> pt {0.0, src_w * -0.5};
	std::vector<float> src_dir = rotate_point(pt, rot_rad);
	std::vector<float> dst_dir = {0.0, dst_w * -0.5};

	std::vector<std::vector<float>> src;
	std::vector<float> tmp, tmp1;
	for (int i = 0; i < center.size(); i++){
		tmp.push_back(center[i] + scale_tmp[i] * shift[i]);
		tmp1.push_back(center[i] + src_dir[i] + scale_tmp[i] * shift[i]);
	}
	// src[0] = center + scale_tmp * shift;
	// src[1] = center + src_dir + scale_tmp * shift;
	
	// src[0] = tmp;
	src.push_back(tmp);
	
	// src[1] = tmp1;
	src.push_back(tmp1);
	// src[2] = _get_3rd_point(src[0], src[1]);
	src.push_back(_get_3rd_point(src[0], src[1]));
	

	
	std::vector<std::vector<float>> dst;
	// dst[0] = {dst_w * 0.5, dst_h * 0.5};
	std::vector<float> tmp2 = {dst_w * 0.5, dst_h * 0.5};
	std::vector<float> tmp3;
	for (int i = 0;i < tmp2.size(); i++){
		tmp3.push_back(tmp2[i] + dst_dir[i]);
	}
	// dst[1] = std::vector<float>({dst_w * 0.5, dst_h * 0.5}) + dst_dir;
	// dst[0] = tmp2;
	// dst[1] = tmp3;
	// dst[2] = _get_3rd_point(dst[0], dst[1]);
	dst.push_back(tmp2);
	dst.push_back(tmp3);
	dst.push_back(_get_3rd_point(dst[0], dst[1]));

	

	cv::Point2f srcTri[3], dstTri[3];
	srcTri[0] = cv::Point2f(src[0][0], src[0][1]);
	srcTri[1] = cv::Point2f(src[1][0], src[1][1]);
	srcTri[2] = cv::Point2f(src[2][0], src[2][1]);

	dstTri[0] = cv::Point2f(dst[0][0], dst[0][1]);
	dstTri[1] = cv::Point2f(dst[1][0], dst[1][1]);
	dstTri[2] = cv::Point2f(dst[2][0], dst[2][1]);

	cv::Mat trans;

	if (inv)
		trans = cv::getAffineTransform(dstTri, srcTri);
	else
		trans = cv::getAffineTransform(srcTri, dstTri);

	return trans;
}


auto top_down_affine(cv::Mat img, std::vector<float> center, std::vector<float> scale, float rotation, 
	std::vector<float> image_size){
	
	// center: [156.06 822.86]
	// scale: [2.3342 3.1123]
	// rotation: 0
	// image_size: [192, 192]
 	
	cv::Mat trans = get_affine_transform(center, scale, rotation, image_size);
	cv::Mat cropped_box;
	// cv::warpAffine(img, cropped_box, trans, img.size());
	cv::warpAffine(img, cropped_box, trans, cv::Size(image_size[0], image_size[1]));
	return cropped_box;
}


void to_tensor(cv::Mat &box){
	// box.convertTo(box, CV_32F);
	// box.convertTo(box, CV_32FC3);
	// box /= 255.0;
	// std::cout << "box: " << box << std::endl;
}


void _get_max_preds(std::vector<std::vector<std::vector<float>>> heatmaps, std::vector<std::vector<float>> &preds, 
	std::vector<std::vector<float>> &max_vals){

	int K = heatmaps.size();
	int W = heatmaps[0][0].size();
	std::vector<std::vector<float>> heatmaps_reshaped;
	std::vector<std::vector<float>> max_vals_tmp;
	for(int i = 0; i < K; i++){
		int max_id = 0;
		float max_val = heatmaps[0][0][0];
		std::vector<float> tmp;
		for(int j = 0; j < heatmaps[0].size(); j++){
			for(int k = 0; k < heatmaps[0][0].size(); k++){
				tmp.push_back(heatmaps[i][j][k]);
				if (heatmaps[i][j][k] > max_val){
					max_val = heatmaps[i][j][k];
					max_id = j * heatmaps[0][0].size() + k; 
				}
		}	
	}

	heatmaps_reshaped.push_back(tmp);
	std::vector<float> pred;
	std::vector<float> max_val_1;
	
	pred.push_back(max_id % W);
	// pred.push_back(ceil(max_id / W));
	pred.push_back(max_id / W);
	preds.push_back(pred);
	max_val_1.push_back(max_val);
	max_vals.push_back(max_val_1);
	max_val_1.push_back(max_val);
	max_vals_tmp.push_back(max_val_1);
	

	}

	for(int i = 0; i < max_vals_tmp.size(); i++){
		for(int j = 0; j < max_vals_tmp[0].size(); j++){
			if (max_vals_tmp[i][j]> 0.0)
				preds[i][j]= preds[i][j];
			else
				preds[i][j] = -1;
			
		}
	}
}


std::vector<float> affine_transform(std::vector<float> pt, cv::Mat trans_mat){
	std::vector<std::vector<float>> trans_vector;
	for (int i = 0; i < 2; i++){
		std::vector<float> tmp;
		for (int j = 0; j < 3; j++){
			tmp.push_back((float)trans_mat.at<double>(i, j));
		}
		trans_vector.push_back(tmp);
	}

	// for(int i = 0; i < trans_vector.size(); i++){
	// 	for(int j = 0; j < trans_vector[0].size(); j++){
	// 		std::cout << trans_vector[i][j] << std::endl;
	// 	}
	// }

	std::vector<float> new_pt;
	for(int i = 0; i < trans_vector.size(); i++)
		new_pt.push_back(trans_vector[i][0] * pt[0] + trans_vector[i][1] * pt[1] + trans_vector[i][2] * 1.0);

	return new_pt; 
}


std::vector<std::vector<float>> transform_preds(std::vector<std::vector<float>> coords, std::vector<float> center, 
	std::vector<float> scale, std::vector<float> output_size){
	std::vector<std::vector<float>> target_coords;
	float rotation = 0.0;
	auto trans = get_affine_transform(center, scale, rotation, output_size, true);
	// std::cout << "post trans: " << trans << std::endl;
	for (int p = 0; p < coords.size(); p++)
		target_coords.push_back(affine_transform(coords[p], trans));


	return target_coords;
}


std::vector<std::vector<float>> keypoints_from_heatmaps(std::vector<std::vector<std::vector<float>>> heatmaps, 
	std::vector<float> center, std::vector<float> scale, bool post_process=true, bool unbiased=false, int kernel=11){

	int K = heatmaps.size();
	float H = heatmaps[0].size();
	float W = heatmaps[0][0].size();
	std::vector<std::vector<float>> preds;
	std::vector<std::vector<float>> max_vals;
	std::vector<std::vector<float>> results;
	_get_max_preds(heatmaps, preds, max_vals);
	


	if (post_process){
		for(int k = 0; k < K; k++){
			auto heatmap = heatmaps[k];
			int px = preds[k][0];
			int py = preds[k][1];

			std::vector<float> diff;
			if (1 < px and px < int(W) - 1 and 1 < py and py < int(H) - 1){
				diff = {heatmap[py][px + 1] - heatmap[py][px - 1], heatmap[py + 1][px] - heatmap[py - 1][px]};
				for(int i = 0; i < diff.size(); i++){
					if (diff[i] > 0.0){
						preds[k][i] +=  0.25;
						// std::cout << "111" << diff[i] << std::endl;
					}else{
						preds[k][i] -=  0.25;
						// std::cout << "222" << diff[i] << std::endl;
						}
					}
						
				} 
			}

		}

	
	
	std::vector<float> output_size {W, H};
	
	preds = transform_preds(preds, center, scale, output_size);

	// for(int i = 0; i < preds.size(); i++){
	// 	for(int j = 0; j < preds[0].size(); j++){
	// 		std::cout << "preds: " << (int)preds[i][j] << std::endl;
	// 	}
	// }

	for(int i = 0; i < preds.size(); i++){
		std::vector<float> tmp;
		tmp.push_back(preds[i][0]);
		tmp.push_back(preds[i][1]);
		tmp.push_back(max_vals[i][0]);
		results.push_back(tmp);
	}

	// std::cout << "ok here" << std::endl;

	// return preds;
	return results;

}


int main(int argc, char** argv) {
	
	const auto poseModel = "./simplebaseline_mv2_256x192.mnn";
	const auto inputImageFileName = "./1.jpg";

	cv::Mat rawImage = cv::imread(inputImageFileName);

	float originalHeight = rawImage.rows;
	float originalWidth = rawImage.cols;
	std::cout << "h, w: " <<originalHeight << " " << originalWidth << std::endl;
	float input_h = 192.0;
	float input_w = 192.0;
	float aspect_ratio_h = originalHeight / input_h;
	float aspect_ratio_w = originalWidth / input_w;
	cv::Mat inputImage = rawImage;
	// cv::resize(rawImage, inputImage, cv::Size(int(input_w), int(input_h)));

	if (inputImage.empty()){
		std::cout << "Empty image! " << std::endl;
		std::cin.get();
		return -1;
	}

	// cv::rectangle(image, cv::Point(230, 40), cv::Point(610, 410), cv::Scalar(0, 255, 0));
	// std::vector<float> box_coords = {230.0 / aspect_ratio_w, 40.0 / aspect_ratio_h, 610.0 / aspect_ratio_w, 410.0 / aspect_ratio_h};
	// std::vector<float> box_coords = {230.0, 40.0, 610.0, 410.0};
	std::vector<float> box_coords = {283.0, 44.0, 499.0, 384.0};
	std::vector<float> center;
	std::vector<float> scale;
	float rotation = 0.0;
	// std::vector<float> image_size = {192.0, 192.0};
	std::vector<float> image_size = {input_h, input_w};

	xyxy2xywh(box_coords);
	box2cs(box_coords, center, scale, image_size);

	inputImage = top_down_affine(inputImage, center, scale, rotation, image_size);

	// to_tensor(inputImage);

	// MNN inference
	auto mnnNet = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(poseModel));
	MNN::ScheduleConfig netConfig;
	netConfig.type = MNN_FORWARD_CPU;
	netConfig.numThread = 4;

	clock_t t1 = clock();
	auto session = mnnNet->createSession(netConfig);
	clock_t t2 = clock();
	std::cout << "createSession: " << (t2 - t1) * 1000 / (double)CLOCKS_PER_SEC << "ms" << std::endl;

	auto input = mnnNet->getSessionInput(session, nullptr);

	mnnNet->resizeTensor(input, {1, 3, input_h, input_w});
	mnnNet->resizeSession(session);

	MNN::CV::ImageProcess::Config config;
	// const float means[3] = {123.675f, 116.28f, 103.53f};
	// const float norms[3] = {0.0171f, 0.0175f, 0.0174f};
	const float means[3] = {123.675, 116.28, 103.53};
	const float norms[3] = {0.0171, 0.0175, 0.0174};
	// const float means[3] = {103.53, 123.675, 116.28};
	// const float norms[3] = {116.28, 123.675, 0.0171};
	// const float means[3] = {0.485, 0.456, 0.406};
	// const float norms[3] = {4.367, 4.464, 4.44};
	// const float norms[3] = {1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225};

	::memcpy(config.mean, means, sizeof(means));
	::memcpy(config.normal, norms, sizeof(norms));
	config.sourceFormat = MNN::CV::BGR;
	config.destFormat = MNN::CV::RGB;
	std::shared_ptr<MNN::CV::ImageProcess> pretreat(
		MNN::CV::ImageProcess::create(config));

	pretreat->convert(inputImage.data, input_w, input_h, inputImage.step[0], input);

	clock_t start = clock();
	mnnNet->runSession(session);
	clock_t end = clock();
	std::cout << "inf time: " << (end - start) * 1000 / (double)CLOCKS_PER_SEC << "ms" << std::endl;

	std::string heatmap_name = "output";
	auto heatmap = mnnNet->getSessionOutput(session, heatmap_name.c_str());

	MNN::Tensor heatmapHost(heatmap, heatmap->getDimensionType());
	heatmap->copyToHostTensor(&heatmapHost);

	std::cout << "Done." << std::endl;
	std::cout << "heatmapHost width: " << heatmapHost.width() << std::endl;
	std::cout << "heatmapHost height: " << heatmapHost.height() << std::endl;
	std::cout << "heatmapHost channel: " << heatmapHost.channel() << std::endl;
	std::cout << "heatmapHost element size: " << heatmapHost.elementSize() << std::endl;


	int batch_size = 1;
	int count = 0;
	std::vector<std::vector<std::vector<float>>> tmp_c;
	for(int i = 0; i < batch_size; i++){
		for(int j = 0; j < 17; j++){
			std::vector<std::vector<float>> tmp_h;
			for(int h = 0; h < 48; h++){
				std::vector<float> tmp_w;
				for(int w = 0; w < 48; w++){
					// std::cout << heatmap->host<float>()[i][j][h][w] << std::endl;
					// tmp_w.push_back(heatmapHost.host<float>()[i * 1 + j * 17 + h * 48 + w * 48]);
					tmp_w.push_back(heatmapHost.host<float>()[i * 1 + j * 48 * 48 + h * 48 + w]);
					if (count < 100)
						std::cout << heatmapHost.host<float>()[i * 1 + j * 48 * 48 + h * 48 + w] << std::endl;
					count += 1;

				}
				tmp_h.push_back(tmp_w);
			}
			tmp_c.push_back(tmp_h);
		}
	}

	auto preds = keypoints_from_heatmaps(tmp_c, center, scale);

	std::vector<std::vector<int>> skeleton = {{16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12},
											{7, 13}, {6, 7}, {6, 8}, {7, 9}, {8, 10}, {9, 11}, {2, 3}, 
											{1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}};

	std::vector<std::vector<int>> palette = {{255, 128, 0}, {255, 153, 51}, {255, 178, 102},
											{230, 230, 0}, {255, 153, 255}, {153, 204, 255},
											{255, 102, 255}, {255, 51, 255}, {102, 178, 255},
											{51, 153, 255}, {255, 153, 153}, {255, 102, 102},
											{255, 51, 51}, {153, 255, 153}, {102, 255, 102},
											{51, 255, 51}, {0, 255, 0}, {0, 0, 255}, {255, 0, 0},
											{255, 255, 255}};

	std::vector<std::vector<int>> pose_limb_color = {palette[0], palette[0], palette[0], palette[0], palette[7], palette[7], palette[7], 
	palette[9], palette[9], palette[9], palette[9], palette[9], palette[16], palette[16], palette[16], palette[16], palette[16], palette[16], palette[16]};

	std::vector<std::vector<int>> pose_kpt_color = {palette[16], palette[16], palette[16], palette[16], palette[16], 
													palette[9], palette[9], palette[9], palette[9], palette[9], palette[9], 
												palette[0], palette[0], palette[0], palette[0], palette[0], palette[0] };


	float kpt_score_thr = 0.3;
	cv::Mat rawImage_copy = rawImage;

	for(int i = 0; i < preds.size(); i++){
		for(int j = 0; j < preds[0].size(); j++){
			std::cout << preds[i][j] << std::endl;
		}
	}

	for(int i = 0; i < preds.size(); i++){
		if (preds[i][2] > kpt_score_thr){
			int r = pose_kpt_color[i][0];
			int g = pose_kpt_color[i][1];
			int b = pose_kpt_color[i][2];
			cv::circle(rawImage_copy, cv::Point(int(preds[i][0]), int(preds[i][1])), 4, cv::Scalar(r, g, b), -1);
		}
		
	}
	
	
	// auto kpts = preds[i];
	auto kpts = preds;
	if(skeleton.size() != 0 and pose_limb_color.size() != 0){
		for(int j = 0; j < skeleton.size(); j++){
			auto sk = skeleton[j];
			std::vector<int> pos1 = {int(kpts[sk[0] - 1][0]), int(kpts[sk[0] - 1][1])};
			std::vector<int> pos2 = {int(kpts[sk[1] - 1][0]), int(kpts[sk[1] - 1][1])};

			if (pos1[0] > 0 and pos1[0] < originalWidth and pos1[1] > 0
				and pos1[1] < originalHeight and pos2[0] > 0 
				and pos2[0] < originalWidth and pos2[1] > 0
				and pos2[1] < originalHeight
				and kpts[sk[0] - 1][2] > kpt_score_thr
				and kpts[sk[1] - 1][2] > kpt_score_thr
				){
				std::vector<int> X = {pos1[0], pos2[0]};
				std::vector<int> Y = {pos1[1], pos2[1]};
				float mX = (X[0] + X[1]) / 2.0;
				float mY = (Y[0] + Y[1]) / 2.0;
				float length = sqrt(pow(Y[0] - Y[1], 2) + pow(X[0] - X[1], 2));
				float angle = atan2(Y[0] - Y[1], X[0] - X[1]) * 180.0 / pi;
				int stickwidth = 2;
				std::vector<cv::Point2i> pts;
				cv::ellipse2Poly(cv::Point2d(int(mX), int(mY)), cv::Size2d(int(length / 2), int(stickwidth)),
					int(angle), 0, 360, 1, pts);

				int r = pose_limb_color[j][0];
				int g = pose_limb_color[j][1];
				int b = pose_limb_color[j][2];
				cv::fillConvexPoly(rawImage_copy, pts, cv::Scalar(r, g, b));
				auto transparency = std::max(0.0, std::min(1.0, 0.5 * (kpts[sk[0] - 1][2] + kpts[sk[1] - 1][2])));

				cv::addWeighted(rawImage_copy, transparency, rawImage, 1 - transparency, 0, rawImage, -1);


			}
		}
	}
	

	std::string imageName = "pose.jpg";
	cv::imwrite(imageName, rawImage);
	std::string windowName = "The Pose";
	cv::namedWindow(windowName);
	cv::imshow(windowName, rawImage);
	cv::waitKey(0);
	cv::destroyWindow(windowName);


	

	// for(int i = 0; i < pose_limb_color.size(); i++){
	// 	for (int j = 0; j < pose_limb_color[0].size(); j++){
	// 		std::cout << pose_limb_color[i][j] << std::endl;
	// 	}
	// }


	

}
