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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <fenv.h>               /*for feenableexcept() */

void allocate_all(double **Vx, double **Vy, double **Vz,
                  double **Fx, double **Fy, double **Fz, int nnodes,
                  double **Xout0, double **Yout0, double **Zout0, int Nout,
                  int *error)
{
    error[0] = 0;
    if ((*Vx = (double *)calloc(nnodes, sizeof(double))) == NULL)
        error[0] += 1; //NOTE error[0]++ does not work
    if ((*Vy = (double *)calloc(nnodes, sizeof(double))) == NULL)
        error[0] += 1;
    if ((*Vz = (double *)calloc(nnodes, sizeof(double))) == NULL)
        error[0] += 1;
    if ((*Fx = (double *)calloc(nnodes, sizeof(double))) == NULL)
        error[0] += 1;
    if ((*Fy = (double *)calloc(nnodes, sizeof(double))) == NULL)
        error[0] += 1;
    if ((*Fz = (double *)calloc(nnodes, sizeof(double))) == NULL)
        error[0] += 1;
    if ((*Xout0 = (double *)calloc(Nout, sizeof(double))) == NULL)
        error[0] += 1;
    if ((*Yout0 = (double *)calloc(Nout, sizeof(double))) == NULL)
        error[0] += 1;
    if ((*Zout0 = (double *)calloc(Nout, sizeof(double))) == NULL)
        error[0] += 1;
}

void deallocate_all(double **Vx, double **Vy, double **Vz,
                    double **Fx, double **Fy, double **Fz,
                    double **Xout0, double **Yout0, double **Zout0)
{
    free(*Vx);
    free(*Vy);
    free(*Vz);
    free(*Fx);
    free(*Fy);
    free(*Fz);
    free(*Xout0);
    free(*Yout0);
    free(*Zout0);
}

void eval_FIRE(double *Vx, double *Vy, double *Vz, double *Fx, double *Fy, double *Fz, int nnodes, double *dt, int init) // PRL 97, 170201 (2006)
{
    const double f_inc = 1.1, f_dec = 0.5, alpha_start = 0.1, f_alpha = 0.99;
    static double alpha, dt_max;
    static int count;
    const int Nmin = 5;
    double totP, modV, modF, temp;
    int i;

    if (!init)
    { //this to assure same initial conditions for every relaxation
        dt_max = 5. * (*dt);
        count = 0;
        alpha = 0.1;
    }

    //F1
    totP = 0.;
    for (i = 0; i < nnodes; i++)
        totP +=
        (
            Fx[i] * Vx[i]
          + Fy[i] * Vy[i]
        //+ Fz[i] * Vz[i]
        );
    //F2-F3
    if (totP > 0.)
    {
        for (i = 0; i < nnodes; i++)
        {
            modF = sqrt(
                  Fx[i] * Fx[i]
                + Fy[i] * Fy[i]
              //+ Fz[i] * Fz[i]
            );
            modV = sqrt(
                  Vx[i] * Vx[i]
                + Vy[i] * Vy[i]
              //+ Vz[i] * Vz[i]
            );
            temp = (modF == 0. ? 0. : alpha * modV / modF);
            Vx[i] = (1. - alpha) * Vx[i] + Fx[i] * temp;
            Vy[i] = (1. - alpha) * Vy[i] + Fy[i] * temp;
          //Vz[i] = (1. - alpha) * Vz[i] + Fz[i] * temp;
        }
        count++;
        if (count > Nmin)
        {
            *dt = ((*dt) * f_inc > dt_max ? dt_max : (*dt) * f_inc);
            alpha *= f_alpha;
        }
    }
    //F4
    else
    { // totP<=0
        for (i = 0; i < nnodes; i++)
            Vx[i] = Vy[i] = Vz[i] = 0.;
        *dt *= f_dec;
        alpha = alpha_start;
        count = 0;
    }
}

void calc_Fmax(double *Fx, double *Fy, double *Fz, int nnodes, double *Fmax)
{
    int i;
    double Fi;
    Fmax[0] = 0.;
    for (i = 0; i < nnodes; i++)
    {
        Fi = (
              Fx[i] * Fx[i]
            + Fy[i] * Fy[i]
          //+ Fz[i] * Fz[i]
        );
        if (Fi > Fmax[0])
            Fmax[0] = Fi;
    }
    Fmax[0] = sqrt(Fmax[0]);
}

void calc_forces(double *X, double *Y, double *Z, int *frozen, int nnodes, int method,
                 int *ipart2, int *jpart2, double *r0, double *kbond, int nbonds,
                 int *ipart3, int *jpart3, int *kpart3, double *a0, double *kangle, int nangles,
                 int *target_in, double *Xin, double *Yin, double *Zin, int Nin,
                 int *target_out, double *Xout, double *Yout, double *Zout,
                 double *Xout0, double *Yout0, double *Zout0, int Nout, double kout,
                 double *Fx, double *Fy, double *Fz, int enforce2D)
{
    double Xij, Yij, Rij, Xik, Yik,/* Zij, Zik, Rik, Rij2, Rik2, RijRik,*/ fxj, fyj, fxk, fyk,/* fzj, fzk,*/ cs, sn, theta, dtheta, radial;
    int i, j, k, n;
    const double twopi=2.*M_PI;

    for (i = 0; i < nnodes; i++)  //resetting forces
        Fx[i] = Fy[i] = Fz[i] = 0.;

    // two-body term
    for (n = 0; n < nbonds; n++)
    {
        if (kbond[n] == 0.)
            continue;
        i = ipart2[n];
        j = jpart2[n];
        Xij = X[j] - X[i];
        Yij = Y[j] - Y[i];
      //Zij = Z[j] - Z[i];
        Rij = sqrt(
              Xij * Xij
            + Yij * Yij
          //+ Zij * Zij
        );

        radial = -kbond[n] * (Rij - r0[n]) / Rij; //NOTE that in fortran code was twice than this

        Fx[i] -= radial * Xij;
        Fy[i] -= radial * Yij;
      //Fz[i] -= radial * Zij;

        Fx[j] += radial * Xij;
        Fy[j] += radial * Yij;
      //Fz[j] += radial * Zij;
    }
    // end of two-body term

    // three-body term
    for (n = 0; n < nangles; n++)
    {
        if (kangle[n] == 0.) continue;
        i = ipart3[n];
        j = jpart3[n];
        k = kpart3[n];

        Xij = X[j] - X[i];
        Yij = Y[j] - Y[i];

        Xik = X[k] - X[i];
        Yik = Y[k] - Y[i];

        sn = (Xij * Yik - Yij * Xik);
        cs = (Xij * Xik + Yij * Yik);

        theta = atan2(sn, cs);                   //tan=sn/cs;
        if ( theta < 0. ) theta += twopi;
        dtheta = theta - a0[n];
	if(dtheta==twopi||dtheta==-twopi) dtheta=0.;             //in case dtheta>=twopi
	//dtheta -= floor(dtheta/twopi)*twopi;

	radial = -kangle[n]*dtheta/(cs*cs+sn*sn); //NOTE that in fortran code it was twice than this

        fxj = radial * (  Yik*cs - Xik*sn );
        fyj = radial * ( -Xik*cs - Yik*sn );
        //fzj = ...

        fxk = radial * ( -Yij*cs - Xij*sn );
        fyk = radial * (  Xij*cs - Yij*sn );
        //fzk = ...

        Fx[i] -= (fxj + fxk);
	Fy[i] -= (fyj + fyk);

	Fx[j] += fxj;
	Fy[j] += fyj;

	Fx[k] += fxk;
	Fy[k] += fyk;

    }
    // end of three-body term

    for (i = 0; i < nnodes; i++){
        if(frozen[i] == 1) Fx[i] = Fy[i] = 0.; //NOTE: input nodes must not be frozen if method==1
        else if(frozen[i] == 2)    Fy[i] = 0.;
        else if(frozen[i] == 3)    Fx[i] = 0.;
    }

    if (method == 0)
    { //displacement-based efficiency
        for (n = 0; n < Nin; n++)
        { //freeze displaced input nodes (may be not needed since they should be frozen)
            i = target_in[n];
            Fx[i] = Fy[i] /*= Fz[i]*/ = 0.;
        }
    }
    else
    { //force-based efficiency
        for (n = 0; n < Nin; n++)
        { //apply external force to input nodes
            i = target_in[n];
            Fx[i] += Xin[n];
            Fy[i] += Yin[n];
          //Fz[i] += Zin[n];
        }
        for (n = 0; n < Nout; n++)
        { //apply spring force to output nodes
            i = target_out[n];

            Xij = X[i] - Xout0[n];
            Fx[i] += -kout * Xij;

            Yij = Y[i] - Yout0[n];
            Fy[i] += -kout * Yij;

          //Zij = Z[i] - Zout0[n];
          //Fz[i] += -kout * Zij;
        }
    }
}

void calc_epot(double *X, double *Y, double *Z, int nnodes, int method,
               int *ipart2, int *jpart2, double *r0, double *kbond, int nbonds,
               int *ipart3, int *jpart3, int *kpart3, double *a0, double *kangle, int nangles,
               //int *target_in, double *Xin, double *Yin, double *Zin, int Nin,
               int *target_out, double *Xout, double *Yout, double *Zout, double *Xout0, double *Yout0, double *Zout0, int Nout, double kout,
               double *Epot)
{
    double Xij, Yij, Rij, Rij2, Xik, Yik,/* Zij, Zik, Rik,*/ theta, dtheta;
    int i, j, k, n;
    const double twopi=2.*M_PI;

    Epot[0] = 0.;

    // two-body term
    for (n = 0; n < nbonds; n++)
    {
        i = ipart2[n];
        j = jpart2[n];

        Xij = X[j] - X[i];
        Yij = Y[j] - Y[i];
      //Zij = Z[j] - Z[i];
        Rij = sqrt(
              Xij * Xij
            + Yij * Yij
          //+ Zij * Zij
        ) - r0[n];

        Epot[0] += 0.5 * kbond[n] * (Rij * Rij); //NOTE that in fortran code it was twice than this
    }

    // three-body term
    for (n = 0; n < nangles; n++)
    {
        i = ipart3[n]; // central
        j = jpart3[n];
        k = kpart3[n];

        Xij = X[j] - X[i];
        Yij = Y[j] - Y[i];
      //Zij = Z[j] - Z[i];

        Xik = X[k] - X[i];
        Yik = Y[k] - Y[i];
      //Zik = Z[k] - Z[i];

        theta = atan2(Xij * Yik - Yij * Xik, Xij * Xik + Yij * Yik);
        if ( theta < 0. ) theta += twopi;;
        dtheta = theta - a0[n];
	if(dtheta==twopi||dtheta==-twopi) dtheta=0.;
	//dtheta -= floor(dtheta/twopi)*twopi; //in case dtheta>=twopi

        Epot[0] += 0.5 * kangle[n] * (dtheta * dtheta); //NOTE that in fortran code it was twice than this

    }

    //external springs in output
    if (method)
    {
        for (n = 0; n < Nout; n++)
        {
            i = target_out[n];
            Xij = X[i] - Xout0[n];
            Yij = Y[i] - Yout0[n];
          //Zij = Z[i] - Zout0[n];
            Rij2 = Xij * Xij + Yij * Yij;// + Zij * Zij;

            Epot[0] += 0.5 * kout * Rij2;
        }
    }
}

void relax(double *X, double *Y, double *Z, int *frozen, int nnodes,                           // coordinates of nodes, frozen flag, array length
           int *ipart2, int *jpart2, double *r0, double *kbond, int nbonds,                    // i,j indexes of bonds, rest length, spring constant, array length
           int *ipart3, int *jpart3, int *kpart3, double *a0, double *kangle, int nangles,     // i,j,k indexes of angles (centered on i), rest angle, spring constant, array length
           int *target_in, double *Xin, double *Yin, double *Zin, int Nin,                     // index of input nodes, direction vector, array length
           int *target_out, double *Xout, double *Yout, double *Zout, int Nout, double kout,   // index of output nodes, direction vector, array length, spring constant in output nodes
           double *Epot, double *Fmax, int method, int enforce2D, int *error, int *step_relax) // potential energy, max force, method (0=displ.-based, 1=force-based), 2D flag, error, relaxation steps
{
    static int once = 1;
    static double *Vx, *Vy, *Vz, *Fx, *Fy, *Fz, *Xout0, *Yout0, *Zout0;
    if (once)
    {
        allocate_all(&Vx, &Vy, &Vz, &Fx, &Fy, &Fz, nnodes, &Xout0, &Yout0, &Zout0, Nout, error);
        if (error[0])
            return;
        once = 0;
    }

    const int max_steps = step_relax[0];
    static const double mass = 1., gamma = 0.; //constants kept to have more explicative v.verlet equations
    double dt = 0.02;                          //timestep is dynamically changed by FIRE
    double halfdt = dt / 2.;// one_plus_ghalfdt = (1. + gamma * dt / 2.);
    int i,n;
    const double Fmax_theshold = Fmax[0];

    if (method == 0)
    { //displacement-based efficiency
        for (n = 0; n < Nin; n++)
        { //displacement of input nodes
            i = target_in[n];
            X[i] += Xin[n];
            Y[i] += Yin[n];
          //Z[i] += Zin[n];
        }
    }
    else
    { //force-based efficiency
        for (n = 0; n < Nout; n++)
        { //saving coords of output nodes
            i = target_out[n];
            Xout0[n] = X[i];
            Yout0[n] = Y[i];
          //Zout0[n] = Z[i];
        }
    }

    calc_forces(X, Y, Z, frozen, nnodes, method,
                ipart2, jpart2, r0, kbond, nbonds,
                ipart3, jpart3, kpart3, a0, kangle, nangles,
                target_in, Xin, Yin, Zin, Nin,
                target_out, Xout, Yout, Zout, Xout0, Yout0, Zout0, Nout, kout,
                Fx, Fy, Fz, enforce2D);

    for (i = 0; i < nnodes; i++)
        Vx[i] = Vy[i] = Vz[i] = 0.; //required for multiple calls of relax()

    //MINIMIZATION LOOP
    for (step_relax[0] = 0; step_relax[0] <= max_steps; step_relax[0] += 1)
    {
        // v( t + dt/2 )
        for (i = 0; i < nnodes; i++)
        {
            Vx[i] += (Fx[i] / mass - gamma * Vx[i]) * halfdt;
            Vy[i] += (Fy[i] / mass - gamma * Vy[i]) * halfdt;
          //Vz[i] += (Fz[i] / mass - gamma * Vz[i]) * halfdt;
        }

        // FIRE changes velocities, it must be called before updating positions
        eval_FIRE(Vx, Vy, Vz, Fx, Fy, Fz, nnodes, &dt, step_relax[0]);

        // FIRE changes dt
        halfdt = dt / 2.;
        //one_plus_ghalfdt = (1. + gamma * halfdt); //commented because gamma=0

        // x( t + dt )
        for (i = 0; i < nnodes; i++)
        {
            X[i] += dt * Vx[i];
            Y[i] += dt * Vy[i];
          //Z[i] += dt * Vz[i];
        }

        // F( t + dt )
        calc_forces(X, Y, Z, frozen, nnodes, method,
                    ipart2, jpart2, r0, kbond, nbonds,
                    ipart3, jpart3, kpart3, a0, kangle, nangles,
                    target_in, Xin, Yin, Zin, Nin,
                    target_out, Xout, Yout, Zout, Xout0, Yout0, Zout0, Nout, kout,
                    Fx, Fy, Fz, enforce2D);

        calc_Fmax(Fx, Fy, Fz, nnodes, Fmax);
        if (Fmax[0] < Fmax_theshold)
            break;

        // v( t + dt )
        for (i = 0; i < nnodes; i++)
        {
            Vx[i] = (Vx[i] + Fx[i] * halfdt / mass);// / one_plus_ghalfdt;
            Vy[i] = (Vy[i] + Fy[i] * halfdt / mass);// / one_plus_ghalfdt;
          //Vz[i] = (Vz[i] + Fz[i] * halfdt / mass);// / one_plus_ghalfdt;
        }
    }

    calc_epot(X, Y, Z, nnodes, method,
              ipart2, jpart2, r0, kbond, nbonds,
              ipart3, jpart3, kpart3, a0, kangle, nangles,
              //target_in, Xin, Yin, Zin, Nin,
              target_out, Xout, Yout, Zout, Xout0, Yout0, Zout0, Nout, kout,
              Epot);
}

int main(int argc, char *argv[])
{
    //feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

    int nnodes = 133, nbonds = 251, nangles = 492, Nin = 10, Nout = 9; //triangular
    //int nnodes = 100, nbonds = 126, nangles = 239, Nin = 3, Nout = 6;  //amorphous
    double X[nnodes], Y[nnodes], Z[nnodes];
    int frozen[nnodes],type[nnodes];
    double X0[nnodes], Y0[nnodes], Z0[nnodes];
    int ipart2[nbonds], jpart2[nbonds];
    double r0[nbonds], kbond[nbonds];
    int ipart3[nangles], jpart3[nangles], kpart3[nangles];
    double a0[nangles], kangle[nangles];
    int target_in[Nin], target_out[Nout];
    double Xin[Nin], Yin[Nin], Zin[Nin], Xout[Nout], Yout[Nout], Zout[Nout];

    int i, j, k, l, m, n, froz;
    double x, y, z, tmp, tmp2, displ;
    char dummy[300],*endp;

    if(argc>1) displ=strtod(argv[1],&endp);
    else displ=sqrt(3.)/8.;
    printf("displacement = %g\n",displ);

    FILE *fnn, *fni, *fno, *fb, *fa;
    if ((fnn = fopen("./input/nodes", "r")) == NULL)
    {
        printf("nodes\n");
        return 1;
    }
    if ((fb = fopen("./input/bonds", "r")) == NULL)
    {
        printf("bonds\n");
        return 1;
    }
    if ((fa = fopen("./input/angles", "r")) == NULL)
    {
        printf("angle\n");
        return 1;
    }
    if ((fni = fopen("./input/nodes_in", "r")) == NULL)
    {
        printf("nodin\n");
        return 1;
    }
    if ((fno = fopen("./input/nodes_out", "r")) == NULL)
    {
        printf("nodou\n");
        return 1;
    }

    m = 0;
    while (1)
    {
        fgets(dummy, sizeof(dummy), fnn);
        if (feof(fnn))
            break;
        if (sscanf(dummy, "%d %d %lg %lg %lg", &n, &l, &x, &y, &z) != 5)
        {
            printf("read1\n");
            return 1;
        }
        n--;
        X0[n] = x;
        Y0[n] = y;
        Z0[n] = z;
	type[n] = l;
        frozen[n] = ( (l==2 || l==3) ? 1 : 0);
        m++;
    }
    printf("read %d nodes\n", m);

    m = 0;
    while (1)
    {
        fgets(dummy, sizeof(dummy), fb);
        if (feof(fb))
            break;
        if (sscanf(dummy, "%d %d %d %d", &n, &l, &i, &j) != 4)
        {
            printf("read4\n");
            return 1;
        }
        n--;i--;j--;
        ipart2[n] = i;
        jpart2[n] = j;
        r0[n] = sqrt( (X0[i]-X0[j])*(X0[i]-X0[j]) + (Y0[i]-Y0[j])*(Y0[i]-Y0[j]) );
        kbond[n] = 10.;
        m++;
    }
    printf("read %d bonds\n", m);

    m = 0;
    while (1)
    {
        fgets(dummy, sizeof(dummy), fa);
        if (feof(fa))
            break;                      //lammps format has j---i---k  (i is central)
        if (sscanf(dummy, "%d %lg %d %d %d %lg", &n, &tmp, &j, &i, &k, &tmp2) != 6)
        {
            printf("read5 %d\n", m);
            return 1;
        }
        n--;i--;j--;k--;
        ipart3[n] = i;
        jpart3[n] = j;
        kpart3[n] = k;
        a0[n] = tmp*M_PI/180.;
        kangle[n] = tmp2;
        m++;
    }
    printf("read %d angles\n", m);

    i = 0;
    while (1)
    {
        fgets(dummy, sizeof(dummy), fni);
        if (feof(fni))
            break;
        if (sscanf(dummy, "%d", &n) != 1)
        {
            printf("read2\n");
            return 1;
        }
        n--;
        target_in[i] = n;
	Xin[i] =  0.;
	Yin[i] = -displ;
	Zin[i] =  0.;
        i++;
    }
    printf("read %d nodes_in\n", i);

    i = 0;
    while (1)
    {
        fgets(dummy, sizeof(dummy), fno);
        if (feof(fno))
            break;
        if (sscanf(dummy, "%d", &n) != 1)
        {
            printf("read3\n");
            return 1;
        }
        n--;
        target_out[i] = n;
        Xout[i] = -1.;
        Yout[i] = 0.;
        Zout[i] = 0.;
        i++;
    }
    printf("read %d nodes_out\n", i);

    fclose(fnn);
    fclose(fni);
    fclose(fno);
    fclose(fb);
    fclose(fa);

    printf("relaxing...\n");

    double Epot, Fmax, kout;
    int error, step_relax;
    int enforce2D = 1;
    int method = 0; // 0=displacement-based, 1=force-based
    char fileout[200];

    for (i = 0; i < 1; i++)
    { // TEST calling relaxation several times

        Fmax = 1e-5;
        kout = 0.00;
        step_relax = 100000;
        error = 0;
        Epot = 0.0;

        for (n = 0; n < nnodes; n++)
        {
            X[n] = X0[n];
            Y[n] = Y0[n];
            Z[n] = Z0[n];
        }

        relax(X, Y, Z, frozen, nnodes,
              ipart2, jpart2, r0, kbond, nbonds,
              ipart3, jpart3, kpart3, a0, kangle, nangles,
              target_in, Xin, Yin, Zin, Nin,
              target_out, Xout, Yout, Zout, Nout, kout,
              &Epot, &Fmax, method, enforce2D, &error, &step_relax);

        if (error)
        {
            printf("error in relaxation.\n");
            return 1;
        }
        printf("%d steps, Epot= %.10lf, Fmax= %.10lf\n", step_relax, Epot, Fmax);

        printf("saving...\n");
        FILE *fout;
        sprintf(fileout, "relaxed_%d", i);
        fout = fopen(fileout, "w");
        if (fout == NULL)
        {
            printf("Error in writing data\n");
            return 1;
        }

        for (j = 0; j < nnodes; j++)
            fprintf(fout, "%d %d %.10lf %.10lf %.10lf\n", j + 1, type[j], X[j], Y[j], Z[j]);

        fclose(fout);
    }
}
