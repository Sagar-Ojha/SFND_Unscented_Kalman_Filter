#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
   is_initialized_ = false;
   time_us_ = 0;
   n_x_ = 5;
   n_aug_ = n_x_ + 2;
   lambda_ = 3 - n_x_;
   weights_ = VectorXd(2*n_aug_ + 1);
   Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

   // If the first loop, then initialize the state and covariance matrices
   if (!is_initialized_)
   {
       // First measurement is assumed to be from Lidar always and in case it's Radar, then the program crashes
       if (meas_package.sensor_type_ == MeasurementPackage::LASER)
       {
           x_(0) = meas_package.raw_measurements_(0);
           x_(1) = meas_package.raw_measurements_(1);
           x_(2) = 0;
           x_(3) = 0;
           x_(4) = 0;

           P_.fill(0.);
           P_(0,0) = std_laspx_*std_laspx_;
           P_(1,1) = std_laspy_*std_laspy_;
           P_(2,2) = 5;
           P_(3,3) = 1;
           P_(4,4) = 1;

           is_initialized_ = true;

           std::cout << "First measurement is from Lidar" << std::endl;
       }
       else
       {
           std::cout << "First measurement is not from Lidar" << std::endl;
       }
   }
   weights_(0) = lambda_/(lambda_+n_aug_);
       for (int i=1; i<2*n_aug_+1; i++)
       {
           weights_(i) = 0.5/(n_aug_+lambda_);
       }

   double delta_t = (meas_package.timestamp_ - time_us_) * 1e-6;
   Prediction(delta_t);
//    std::cout << delta_t << std::endl;

   // Check if the sensor tyoe is LASER or RADAR
   if (meas_package.sensor_type_ == MeasurementPackage::LASER)
   {
//        std::cout << "Lidar measurement" << std::endl;
       UpdateLidar(meas_package);
   }
   else
   {
//        std::cout << "Radar measurement" << std::endl;
       UpdateRadar(meas_package);
   }
   time_us_ = meas_package.timestamp_;

   if (std::isnan(x_(0)) || std::isnan(x_(1)) || std::isnan(x_(2))) {
       std::cout << "NaN detected in state vector!" << std::endl;
       while (true)
       {
       }
    }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

   // Augment the state vector
   VectorXd x_aug = VectorXd(n_aug_);
   x_aug.head(n_x_) = x_;
   x_aug(n_x_) = 0;
   x_aug(n_x_+1) = 0;

   // Augment the P matrix
   MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
   P_aug.fill(0.);
   P_aug.topLeftCorner(n_x_, n_x_) = P_;
   P_aug(n_x_, n_x_) = std_a_*std_a_;
   P_aug(n_x_+1, n_x_+1) = std_yawdd_*std_yawdd_;

   // create square root matrix
   Eigen::LLT<MatrixXd> llt(P_aug);
   if(llt.info() == Eigen::NumericalIssue)
   {
     std::cout << "P_aug is not positive semi-definite" << std::endl;
     P_aug += Eigen::MatrixXd::Identity(n_aug_, n_aug_) * 1e-5;	// Regularization
     llt = Eigen::LLT<MatrixXd>(P_aug);
   }
   MatrixXd sqrtMat = llt.matrixL();

   // Sigma point matrix
   MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

   // create augmented sigma points
   Xsig_aug.col(0) = x_aug;
   Eigen::MatrixXd x_augPlusMinus = sqrt(lambda_ + n_aug_) * sqrtMat;

   for(int i=1; i<=n_aug_; i++)
   {
     Xsig_aug.col(i) = x_aug + x_augPlusMinus.col(i-1);
     Xsig_aug.col(i+n_aug_) = x_aug - x_augPlusMinus.col(i-1);
   }

   // predict sigma points
   for (int i=0; i<(2*n_aug_+1); i++)
   {
       double v = Xsig_aug.col(i)(2);
       double psi = Xsig_aug.col(i)(3);
       double psiDot = Xsig_aug.col(i)(4);

       // Noise states might become non zero after unscented transform
       double nuA = Xsig_aug.col(i)(5);
       double nuPsiDDot = Xsig_aug.col(i)(6);

       VectorXd noiseTerm(n_x_);
       noiseTerm(0) = 0.5*delta_t*delta_t*cos(psi)*nuA;
       noiseTerm(1) = 0.5*delta_t*delta_t*sin(psi)*nuA;
       noiseTerm(2) = delta_t*nuA;
       noiseTerm(3) = 0.5*delta_t*delta_t*nuPsiDDot;
       noiseTerm(4) = delta_t*nuPsiDDot;

	   VectorXd processDeterministicTerm = VectorXd(n_x_);

       if (psiDot != 0)
       {
           processDeterministicTerm(0) = (v/psiDot) * (sin(psi+psiDot*delta_t) - sin(psi));
           processDeterministicTerm(1) = (v/psiDot) * (-cos(psi+psiDot*delta_t) + cos(psi));
           processDeterministicTerm(2) = 0;
           processDeterministicTerm(3) = psiDot*delta_t;
           processDeterministicTerm(4) = 0;

           Xsig_pred_.col(i) = Xsig_aug.col(i).head(n_x_) + processDeterministicTerm + noiseTerm;
//            std::cout << "---------------" << std::endl;
       }
       else
       {
           processDeterministicTerm(0) = v * cos(psi)*delta_t;
           processDeterministicTerm(1) = v * sin(psi)*delta_t;
           processDeterministicTerm(2) = 0;
           processDeterministicTerm(3) = psiDot*delta_t;
           processDeterministicTerm(4) = 0;

           Xsig_pred_.col(i) = (Xsig_aug.col(i)).head(n_x_) + processDeterministicTerm + noiseTerm;
//            std::cout << "----------+++++" << std::endl;
       }
   }
//    std::cout << "------------------------------" << std::endl;

   VectorXd x = VectorXd::Zero(n_x_);

   for (int i=0; i < (2*n_aug_ + 1); i++)
   {
       x += weights_(i) * Xsig_pred_.col(i);
   }
   x_ = x;

   MatrixXd P = MatrixXd::Zero(n_x_, n_x_);
   for (int i=0; i < (2*n_aug_ + 1); i++)
   {
       // predict state covariance matrix
       P += weights_(i) * (Xsig_pred_.col(i) - x_) * ((Xsig_pred_.col(i) - x_).transpose());
   }
   P_ = P;

   return;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

   // Estimated output
   int n_z = 2;
   MatrixXd H = MatrixXd(n_z, n_x_);
   H << 1, 0, 0, 0, 0,
  		0, 1, 0, 0, 0;

   VectorXd y = meas_package.raw_measurements_ - H*x_;
  
   // Measurement Covariance
   MatrixXd R = MatrixXd(n_z, n_z);
   R << std_laspx_*std_laspx_, 0,
        0, std_laspy_*std_laspy_;
  
   // Kalman Gain
   MatrixXd S = H * P_ * (H.transpose()) + R;
   MatrixXd K = P_ * (H.transpose()) * (S.inverse());
   
   x_ += K * y;
   MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
   P_  -=  K * H * P_;
  
   // TODO: Try Unscented Transform for LIDAR
   // Augmented output
//    int n_z = 2;
//    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
//    for (int i=0; i < (2 * n_aug_ + 1); i++)
//    {
//        Zsig.col(i)(0) = Xsig_pred_.col(i)(0);
//        Zsig.col(i)(1) = Xsig_pred_.col(i)(1);
//    }

//    // calculate mean predicted measurement
//    VectorXd z_pred = VectorXd(n_z);
//    for (int i=0; i < (2*n_aug_+1); i++)
//    {
//        z_pred += weights_(i) * Zsig.col(i);
//    }

//    // calculate innovation covariance matrix S
//    MatrixXd R = MatrixXd(n_z, n_z);
//    R << std_laspx_*std_laspx_, 0,
//         0, std_laspy_*std_laspy_;
//    MatrixXd S = R;

//    for (int i=0; i < (2*n_aug_+1); i++)
//    {
//        S += weights_(i) * (Zsig.col(i) - z_pred) * ((Zsig.col(i) - z_pred).transpose());
//    }

//    // calculate cross correlation matrix
//    MatrixXd Tc = MatrixXd(n_x_, n_z);
//    for (int i=0; i<(2*n_aug_ + 1); i++)
//    {
//        Tc += weights_(i) * (Xsig_pred_.col(i) - x_) * ((Zsig.col(i) - z_pred).transpose());
//    }

//    // calculate Kalman gain K
//    MatrixXd K = Tc * (S.inverse());

//    // update state mean and covariance matrix
//    x_ += K * (meas_package.raw_measurements_ - z_pred);
//    P_ -= K * S * (K.transpose());

   return;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

   // Augmented output
   int n_z = 3;
   MatrixXd Zsig = MatrixXd::Zero(n_z, 2 * n_aug_ + 1);

   // transform sigma points into measurement space
   for (int i=0; i < (2 * n_aug_ + 1); i++)
   {
       Zsig.col(i)(0) = sqrt(pow(Xsig_pred_.col(i)(0), 2) + pow(Xsig_pred_.col(i)(1), 2));
       Zsig.col(i)(1) = atan2(Xsig_pred_.col(i)(1), Xsig_pred_.col(i)(0));
       if (Zsig.col(i)(0) != 0)
       {
       	   Zsig.col(i)(2) = (Xsig_pred_.col(i)(0) * cos(Xsig_pred_.col(i)(3)) * Xsig_pred_.col(i)(2) + Xsig_pred_.col(i)(1) * sin(Xsig_pred_.col(i)(3)) * Xsig_pred_.col(i)(2)) / Zsig.col(i)(0);
       }
       else
       {
           Zsig.col(i)(2) = 0.;
           std::cout << "Radial distance is too small" << std::endl;
       }
   }

   // calculate mean predicted measurement
   VectorXd z_pred = VectorXd::Zero(n_z);
   for (int i=0; i < (2*n_aug_+1); i++)
   {
       z_pred += weights_(i) * Zsig.col(i);
   }

   // calculate innovation covariance matrix S
   MatrixXd R = MatrixXd(n_z, n_z);
   R << std_radr_*std_radr_, 0, 0,
        0, std_radphi_*std_radphi_, 0,
        0, 0, std_radrd_*std_radrd_;
   MatrixXd S = R;

   for (int i=0; i < (2*n_aug_+1); i++)
   {
       S += weights_(i) * (Zsig.col(i) - z_pred) * ((Zsig.col(i) - z_pred).transpose());
   }

   // calculate cross correlation matrix
   MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
   for (int i=0; i<(2*n_aug_ + 1); i++)
   {
       Tc += weights_(i) * (Xsig_pred_.col(i) - x_) * ((Zsig.col(i) - z_pred).transpose());
   }

   // calculate Kalman gain K
   MatrixXd K = Tc * (S.inverse());

   // update state mean and covariance matrix
   x_ += K * (meas_package.raw_measurements_ - z_pred);
   P_ -= K * S * (K.transpose());

   return;
}