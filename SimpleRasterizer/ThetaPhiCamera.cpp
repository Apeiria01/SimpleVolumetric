#include "ThetaPhiCamera.h"
#include <Eigen/Core>

ThetaPhiCamera::ThetaPhiCamera()
{
	m_cpuView = Eigen::Matrix4f::Identity();
	m_translation= Eigen::Array3f(0.0f, 0.0f, -4.0f);
	m_xyzRotation = Eigen::Array3f(0.0f, 0.0f, 0.0f);
	m_xyzScale = Eigen::Array3f(1.0f, 1.0f, 1.0f);
	UpdateSpin();
}

const Eigen::Matrix4f& ThetaPhiCamera::GetMat() {
	return m_cpuView;
}

void ThetaPhiCamera::UpdateSpin() {
	Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
	m_cpuView = I;
	Eigen::Matrix4f rotation = Eigen::Matrix4f::Identity();
	rotation <<
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, cos(-m_xyzRotation[0]), -sin(-m_xyzRotation[0]), 0.0f,
		0.0f, sin(-m_xyzRotation[0]), cos(-m_xyzRotation[0]), 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f;
	m_cpuView = rotation * m_cpuView;
	Eigen::Matrix4f rotation2 = Eigen::Matrix4f::Identity();
	
	rotation2 <<
		cos(m_xyzRotation[1]), 0.0f, -sin(m_xyzRotation[1]), 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		sin(m_xyzRotation[1]), 0.0f, cos(m_xyzRotation[1]), 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f;
	//m_cpuView = invRot * m_cpuView;
	m_cpuView = rotation2 * m_cpuView;
	Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
	Eigen::Vector4f realPos = Eigen::Vector4f(m_translation[0], m_translation[1], m_translation[2], 0.0f);
	realPos = m_cpuView * realPos;
	translation <<
		1.0f, 0.0f, 0.0f, realPos[0],
		0.0f, 1.0f, 0.0f, realPos[1],
		0.0f, 0.0f, 1.0f, realPos[2],
		0.0f, 0.0f, 0.0f, 1.0f;
	m_cpuView = translation * m_cpuView;
	Eigen::Matrix4f scale = Eigen::Matrix4f::Identity();
	scale <<
		m_xyzScale[0], 0.0f, 0.0f, 0.0f,
		0.0f, m_xyzScale[1], 0.0f, 0.0f,
		0.0f, 0.0f, m_xyzScale[2], 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f;
	m_cpuView = scale * m_cpuView;
	return;
}

void ThetaPhiCamera::CalcThetaPhiSpin(int dx, int dy) {
	float dxTheta = dx / 64.0f;
	float dyTheta = dy / 64.0f;
	m_xyzRotation += Eigen::Array3f(dyTheta, dxTheta, 0.0f);
	UpdateSpin();
}

void ThetaPhiCamera::CalcScale(int sc) {
	m_xyzScale -= sc * Eigen::Array3f(0.0005f, 0.0005f, 0.0005f);
	if(m_xyzScale[0] <= 0.1f) m_xyzScale = m_xyzScale = Eigen::Array3f(0.1f, 0.1f, 0.1f);
	if(m_xyzScale[0] >= 5.0f) m_xyzScale = m_xyzScale = Eigen::Array3f(5.0f, 5.0f, 5.0f);
	UpdateSpin();
}