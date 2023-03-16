#pragma once
#include <Eigen/Core>

class ThetaPhiCamera {
private:
	int m_width;
	int m_height;
	Eigen::Array3f m_xyzRotation;
	Eigen::Array3f m_translation;
	Eigen::Array3f m_xyzScale;
	Eigen::Matrix4f m_cpuView;

public:
	const Eigen::Matrix4f& GetMat();
	ThetaPhiCamera(int width, int height);
	void UpdateSpin();
	void CalcThetaPhiSpin(int dx, int dy);
	void CalcScale(int sc);
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};