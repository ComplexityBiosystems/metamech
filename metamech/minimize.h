//  Codes used for the paper: 
//  "Automatic Design of Mechanical Metamaterial Actuators" 
//  by S. Bonfanti, R. Guerra, F. Font-Clos, R. Rayneau-Kirkhope, S. Zapperi
//  Center for Complexity and Biosystems, University of Milan
//  (c) University of Milan 
// 
// 
// #####################################################################
// 
//  End User License Agreement (EULA)
//  Your access to and use of the downloadable code (the "Code") is subject
//  to a non-exclusive,  revocable, non-transferable,  and limited right to
//  use the Code for the exclusive purpose of undertaking academic, 
//  governmental, or not-for-profit research. Use of the Code or any part
//  thereof for commercial purposes is strictly prohibited in the absence
//  of a Commercial License Agreement from the University of Milan. For 
//  information contact the Technology Transfer Office of the university
//  of Milan (email: tto@unimi.it)
// 
// ######################################################################

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void allocate_all(double **Vx, double **Vy, double **Vz,
                  double **Fx, double **Fy, double **Fz, int nnodes,
                  double **Xout0, double **Yout0, double **Zout0, int Nout,
                  int *error);
void eval_FIRE(double *Vx, double *Vy, double *Vz, double *Fx, double *Fy, double *Fz, int nnodes, double &dt);
void calc_Fmax(double *Fx, double *Fy, double *Fz, int nnodes, double *Fmax);
void calc_forces(double *X, double *Y, double *Z, int *frozen, int nnodes, int method,
               int *ipart2, int *jpart2, double *r0, double *kbond, int nbonds,
               int *ipart3, int *jpart3, int *kpart3, double *a0, double *kangle, int nangles,
               int *target_in, double *Xin, double *Yin, double *Zin, int Nin,
               int *target_out, double *Xout, double *Yout, double *Zout,
               double *Xout0, double *Yout0, double *Zout0, int Nout, double kout,
               double *Fx, double *Fy, double *Fz, int enforce2D);
void calc_epot(double *X, double *Y, double *Z, int nnodes, int method,
               int *ipart2, int *jpart2, double *r0, double *kbond, int nbonds,
               int *ipart3, int *jpart3, int *kpart3, double *a0, double *kangle, int nangles,
               //int *target_in, double *Xin, double *Yin, double *Zin, int Nin,
               int *target_out, double *Xout, double *Yout, double *Zout, double *Xout0, double *Yout0, double *Zout0, int Nout, double kout,
               double *Epot);
void relax(double *X, double *Y, double *Z, int *frozen, int nnodes,                                 // coordinates of nodes, frozen flag, array length
           int *ipart2, int *jpart2, double *r0, double *kbond, int nbonds,                          // i,j indexes of bonds, rest length, spring constant, array length
           int *ipart3, int *jpart3, int *kpart3, double *a0, double *kangle, int nangles,           // i,j,k indexes of angles (centered on i), rest angle, spring constant, array length
           int *target_in, double *Xin, double *Yin, double *Zin, int Nin,                           // index of input nodes, direction vector, array length
           int *target_out, double *Xout, double *Yout, double *Zout, int Nout, double kout,         // index of output nodes, direction vector, array length, spring constant in output nodes
           double *Epot, double *Fmax, int method, int enforce2D, int *error, int *step_relax);
           
            
#ifdef __cplusplus
}
#endif