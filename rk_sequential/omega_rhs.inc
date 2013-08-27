/************************ OMEGA_RHS.INC ******************************/
void rhs(// BH masses
         const double m1,
	 // BH spins
	 const double chi1, const double chi2,
         // state: note all vectors are unit vectors
         const double omega,
	 const double S1ux, const double S1uy, const double S1uz,
	 const double S2ux, const double S2uy, const double S2uz,
	 const double LNx,  const double LNy,  const double LNz,
         // rhs
         double *domega,
	 double *dS1ux, double *dS1uy, double *dS1uz,
	 double *dS2ux, double *dS2uy, double *dS2uz,
	 double *dLNx,  double *dLNy,  double *dLNz)
{
  // shorthands
  const double om=omega;

  // mass constants: total mass M=m1+m2=1.0(!)
  const double m2=1.0-m1;
  // eta=m1*m2/M^2=m1*m2 for M=1
  const double eta=m1*m2;
  const double eta_2=eta*eta;

  // performance abbrevations (minimize pow() function calls)
  const double om_2=om*om;
  const double m1_2=m1*m1;
  const double m2_2=m2*m2;
  const double om_1_3=pow(om,1./3.);
  const double om_2_3=om_1_3*om_1_3;
  const double om_4_3=om_2_3*om_2_3;
  const double om_5_3=om_4_3*om_1_3;
  const double om_7_3=om_5_3*om_2_3;
  const double om_11_3=om_7_3*om_4_3;

  // convert spin magnitude
  const double u1=chi1*m1_2;
  const double u2=chi2*m2_2;
  const double iu1=1./u1;
  const double iu2=1./u2;

  // actual spin vectors (i.e. not unit)
  const double S1x=u1*S1ux;
  const double S1y=u1*S1uy;
  const double S1z=u1*S1uz;

  const double S2x=u2*S2ux;
  const double S2y=u2*S2uy;
  const double S2z=u2*S2uz;

  // assign rhs quantities
  *domega=19.2*eta*om_11_3*(-om*(.08333333333333333*(chi2*(113.0*
   (m2*m2)+75.0*eta)*(LNz*S2uz+LNy*S2uy+LNx*S2ux)+chi1*(113.0*(m1*m1)+
   75.0*eta)*(LNz*S1uz+LNy*S1uy+LNx*S1ux))-12.56637061435917)-
   .02083333333333333*chi1*chi2*eta*om_4_3*(247.0*(S1uz*S2uz+S1uy*S2uy+
   S1ux*S2ux)-721.0*(LNz*S1uz+LNy*S1uy+LNx*S1ux)*(LNz*S2uz+LNy*S2uy+LNx
   *S2ux))+3.141592653589793*(60.51256613756614*eta_2+
   59.30472883597884*eta-1.094990079365079)*om_7_3-.004674989067841953*
   (15876.0*eta+4159.0)*om_5_3+(3.277777777777778*eta_2+
   6.776289682539683*eta+1.879574514991182)*om_4_3+om_2*
   (-8.152380952380952*log(16.0*om_2_3)+.6037946428571429*eta_2-
   2.162422839506173*(eta*eta*eta)-165.381044105182*eta+
   160.9522690075005)-.002976190476190476*(924.0*eta+743.0)*om_2_3+1.0);

  // rhs for spins
  const double dS1x=0.5*om_2*S1z*(-3.0*LNy*(LNz*S2z+LNy*S2y+LNx*S2x)+
	S2y+eta*LNy*(3.0*m2/m1+4.0)/om_1_3)-0.5*om_2*S1y*(-3.0*LNz*(LNz*S2z+
	LNy*S2y+LNx*S2x)+S2z+eta*LNz*(3.0*m2/m1+4.0)/om_1_3);

  const double dS1y=0.5*om_2*S1x*(-3.0*LNz*(LNz*S2z+LNy*S2y+LNx*S2x)+
	S2z+eta*LNz*(3.0*m2/m1+4.0)/om_1_3)-0.5*om_2*S1z*(-3.0*LNx*(LNz*S2z+
	LNy*S2y+LNx*S2x)+S2x+eta*LNx*(3.0*m2/m1+4.0)/om_1_3);

  const double dS1z=0.5*om_2*S1y*(-3.0*LNx*(LNz*S2z+LNy*S2y+LNx*S2x)+
	S2x+eta*LNx*(3.0*m2/m1+4.0)/om_1_3)-0.5*om_2*S1x*(-3.0*LNy*(LNz*S2z+
	LNy*S2y+LNx*S2x)+S2y+eta*LNy*(3.0*m2/m1+4.0)/om_1_3);

  const double dS2x=0.5*om_2*(-3.0*LNy*(LNz*S1z+LNy*S1y+LNx*S1x)+S1y+
	eta*LNy*(3.0*m1/m2+4.0)*1.0/om_1_3)*S2z-0.5*om_2*(-3.0*LNz*(LNz*S1z+
	LNy*S1y+LNx*S1x)+S1z+eta*LNz*(3.0*m1*1.0/m2+4.0)/om_1_3)*S2y;

  const double dS2y=0.5*om_2*(-3.0*LNz*(LNz*S1z+LNy*S1y+LNx*S1x)+S1z+
	eta*LNz*(3.0*m1/m2+4.0)*1.0/om_1_3)*S2x-0.5*om_2*(-3.0*LNx*(LNz*S1z+
	LNy*S1y+LNx*S1x)+S1x+eta*LNx*(3.0*m1*1.0/m2+4.0)/om_1_3)*S2z;

  const double dS2z=0.5*om_2*(-3.0*LNx*(LNz*S1z+LNy*S1y+LNx*S1x)+S1x+
	eta*LNx*(3.0*m1/m2+4.0)*1.0/om_1_3)*S2y-0.5*om_2*(-3.0*LNy*(LNz*S1z+
	LNy*S1y+LNx*S1x)+S1y+eta*LNy*(3.0*m1*1.0/m2+4.0)/om_1_3)*S2x;

  // rhs for unit spin vectors
  *dS1ux=dS1x*iu1;
  *dS1uy=dS1y*iu1;
  *dS1uz=dS1z*iu1;

  *dS2ux=dS2x*iu2;
  *dS2uy=dS2y*iu2;
  *dS2uz=dS2z*iu2;

  // rhs for unit angular momentum vectors
  *dLNx=-1.0*om_1_3*(dS2x+dS1x)/eta;

  *dLNy=-1.0*om_1_3*(dS2y+dS1y)/eta;

  *dLNz=-1.0*om_1_3*(dS2z+dS1z)/eta;

} // end of rhs()
