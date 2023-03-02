#pragma once
#include <Eigen/Core>

class ThetaPhiCamera {
private:
	Eigen::Array3f m_xyzRotation;
	Eigen::Array3f m_translation;
	Eigen::Array3f m_xyzScale;
	Eigen::Matrix4f m_cpuView;

public:
	const Eigen::Matrix4f& GetMat();
	ThetaPhiCamera();
	void UpdateSpin();
	void CalcThetaPhiSpin(int dx, int dy);
	void CalcScale(int sc);
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};