#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

static int	csfcn(int num, double (*zbk)[3], double eta[3], double *suf);
static double	dsbp(int *nvs, int ris[4], int rjs[4], double y[4][3],
			double dell[4][4], double zsol[3],
			double als[3], int *backup);

void gilbert(int nvi, int nvj, double (*zi)[3], double (*zj)[3], double zisol[3], double zjsol[3])
{
	int	nvs;		/* number of points in polytope set to zero
				** on initialization */
	int	ris[4];		/* index to vertices in polytope I */
	int	rjs[4];		/* index to vertices in polytope J */
	double	als[4];		
	double nzsol[3];
	int ri,rj,i,j,k,l,ii,jj,kk,ll;
	double y[4][3],oldy[4][3];
	double dell[4][4],olddell[4][4];
	int oldris[4], oldrjs[4], iord[4];
	double zsol[3];
	double lastdstsq, dstsq, si, sj;
	int backup=0;
	int oldnvs,ncy;

	ncy=0;
	nvs=1;
	ris[0]=0;
	rjs[0]=0;
	als[0]=1.0;

	/* update polytope and dot product table */
	for (i=0; i < nvs; i++) {
		ii = ris[i];
		jj = rjs[i];
		y[i][0] = zi[ii][0] - zj[jj][0];
		y[i][1] = zi[ii][1] - zj[jj][1];
		y[i][2] = zi[ii][2] - zj[jj][2];
	}
	for (i=0; i < nvs; i++) {
		for (j=0; j<i; j++) {
			dell[i][j] = y[i][0]*y[j][0]
			    + y[i][1]*y[j][1]
			    + y[i][2]*y[j][2];
		}
		dell[i][i] = y[i][0]*y[i][0] + y[i][1]*y[i][1] + y[i][2]*y[i][2];
	}
	lastdstsq=dell[0][0]+dell[0][0]+1.0;
	for (;;) {
		ncy++;

		/* call distance sub algorythm */


		dstsq=dsbp(&nvs,ris,rjs,
		    y,dell,zsol,als,&backup);

		if (dstsq >= lastdstsq || nvs == 4) {
			if (backup) {

				for (i = 0; i < 3; i++) {
					zisol[i] = zjsol[i] = 0.0;
					for (j = 0; j < nvs; j++) {
						zisol[i] += zi[ris[j]][i]*als[j];
						zjsol[i] += zj[rjs[j]][i]*als[j];
					
					}
				}

				/* make sure intersection has zero distance */
				if (nvs == 4) {
					for (i = 0; i < 3; i++)
						zisol[i] = zjsol[i] =
							0.5*(zisol[i]+zjsol[i]);
				}

				return;
			}
			backup=1;
			if (ncy == 1) continue;
			nvs = oldnvs;
			for (k=0;k<nvs;k++) {
				ris[k] = oldris[k];
				rjs[k] = oldrjs[k];
				y[k][0] = oldy[k][0];
				y[k][1] = oldy[k][1];
				y[k][2] = oldy[k][2];
				for (l=0;l<k;l++) {
					dell[k][l]=olddell[k][l];
				}
				dell[k][k]=olddell[k][k];
			}
			continue;
		}
		lastdstsq=dstsq;
		/* find new point to add to polytope */
		nzsol[0]= -zsol[0];
		nzsol[1]= -zsol[1];
		nzsol[2]= -zsol[2];
		ri=csfcn(nvi,zi,nzsol,&si);
		rj=csfcn(nvj,zj,zsol,&sj);
		/* if not add new point to polytope and try again */
		/* move first point to last position */
		ris[nvs]=ris[0];
		rjs[nvs]=rjs[0];
		y[nvs][0] = y[0][0];
		y[nvs][1] = y[0][1];
		y[nvs][2] = y[0][2];
		for (i=0;i<nvs;i++) {
			dell[nvs][i] = dell[i][0];
		}
		dell[nvs][nvs]=dell[0][0];

		/* put new point in first spot */

		ris[0]=ri;
		rjs[0]=rj;
		y[0][0] = zi[ri][0] - zj[rj][0];
		y[0][1] = zi[ri][1] - zj[rj][1];
		y[0][2] = zi[ri][2] - zj[rj][2];
		/* update dot product table */
		for (i=0;i<=nvs;i++) {
			dell[i][0] = y[i][0]*y[0][0]
			    + y[i][1]*y[0][1]
			    + y[i][2]*y[0][2];
		}
		nvs++;

		/* save old values of nvs, ris, rjs, y and dell */

		oldnvs=nvs;
		for (k=0;k<nvs;k++) {
			oldris[k]=ris[k];
			oldrjs[k]=rjs[k];
			oldy[k][0] = y[k][0];
			oldy[k][1] = y[k][1];
			oldy[k][2] = y[k][2];
			for (l=0;l<k;l++) {
				olddell[k][l]=dell[k][l];
			}
			olddell[k][k]=dell[k][k];
		}
		/* if nvs == 4, 
		** rearrange dell[1][0], dell[2][1] and dell[3][0]
		** in non decreasing order
		*/

		if (nvs == 4) {
			iord[0] = 0;
			iord[1] = 1;
			iord[2] = 2;
			if (dell[2][0] < dell[1][0]) {
				iord[1] = 2;
				iord[2] = 1;
			}
			ii = iord[1];
			if (dell[3][0] < dell[ii][0]) {
				iord[3] = iord[2];
				iord[2] = iord[1];
				iord[1] = 3;
			} else {
				ii = iord[2];
				if (dell[3][0] < dell[ii][0]) {
					iord[3] = iord[2];
					iord[2] = 3;
				} else {
					iord[3] = 3;
				}
			}
			/* reorder ris,rjs y and dell */
			for (k=1;k<nvs;k++) {
				kk = iord[k];
				ris[k] = oldris[kk];
				rjs[k] = oldrjs[kk];
				y[k][0] = oldy[kk][0];
				y[k][1] = oldy[kk][1];
				y[k][2] = oldy[kk][2];
				for (l=0;l<k;l++) {
					ll = iord[l];
					if (kk >= ll)
						dell[k][l] = olddell[kk][ll];
					else
						dell[k][l] = olddell[ll][kk];
				}
				dell[k][k] = olddell[kk][kk];
			}
		}
	}
}


static int
csfcn(int num, double (*zbk)[3], double eta[3], double *suf)
{
	int i;
	double max,t;
	int index;

	index = 0;
	max = zbk[0][0]*eta[0] + zbk[0][1]*eta[1] + zbk[0][2]*eta[2];
	for (i=1; i < num; i++) {
		t = zbk[i][0]*eta[0] + zbk[i][1]*eta[1] + zbk[i][2]*eta[2];
		if (t > max) {
			index = i;
			max = t;
		}
	}
	*suf=max;
	return index;
}




double
dsbp(int *nvs, int ris[4], int rjs[4], double y[4][3],
	double dell[4][4], double zsol[3], double als[3], int *backup)
/*
**
**  dsbp implements, in a very efficient way, the distance subalgorithm
**  of finding the near point to the convex hull of four or less points
**  in 3-D space. The procedure and its efficient FORTRAN implementation
**  are both due to D.W.Johnson. Although this subroutine is quite long,
**  only a very small part of it will be executed on each call. Refer to
**  sections 5 and 6 of the report mentioned in routine DIST3 for details
**  concerning the distance subalgorithm. Converted to C be Diego C. Ruspini
**  3/25/93.
**
**  Following is a brief description of the parameters in dsbp :
**
**  **** on input :
**
**  *nvs           :  The number of points.  1 <= *nvs <= 4 .
**
**  y[index][d]    :  The array containing the points.
**
**  ris[*], rjs[*] :  Index vectors for Polytope-I and Polytope-J.
**                    For k = 1,...,nvs, 
**                        y[k] = zbi[ris[k]] - zbj[rjs[k]].
**
**  dell[*][*]      :  dell[i][j] = Inner product of y[i] and y[j].
**
**  **** ON  OUTPUT : 
**
**  zsol[*]        :  Near point to the convex hull of the points 
**                    in y.
**
**  **** dsbp also determines an affinely independent subset of the
**  points such that zsol= near point to the affine hull of the points
**  in the subset. The variables nvs, y, ris, rjs and dell are modified
**  so that, on output, they correspond to this subset of points.
**
**  als[*]         :  The barycentric coordinates of zsol, i.e.,
**                    zsol = als[0]*y[1] + ... + ALS(nvs)*y[nvs-1],
**                    als[k] > 0.0 for k=0,...,nvs-1, and,
**                    als[0] + ... + als[nvs-1] = 1.0 .
**                       
**
*/
{
	int k,l,kk,ll;
	int nvsd;
	int risd[4],rjsd[4],iord[4];
	double sum;
	double e132,e142,e123,e143,e213,e243;
	double e124,e134,e214,e234,e314,e324;
	double d1[15],d2[15],d3[15],d4[15];
	double yd[3][4], delld[4][4], zsold[3];
	double dstsq;
	double alsd[4], dstsqd;


	d1[0]=d2[1]=d3[3]=d4[7]=1.0;
	if (!*backup) {
		/* regular distance subalgoritm begins */
		switch (*nvs) {
		case 1: 
			{	/* case  of  a  single  point ... */
				als[0] = d1[0];
				zsol[0] = y[0][0];
				zsol[1] = y[0][1];
				zsol[2] = y[0][2];
				dstsq = dell[0][0];
				return(dstsq);
				break;
			} /* END case one point */
		case 2: 
			{	/* case of two points ... */
				/* check optimality of vetex 1 */
				d2[2] = dell[0][0] - dell[1][0];
				if (d2[2] <= 0.0) {
					*nvs = 1;
					als[0] = d1[0];
					zsol[0] = y[0][0];
					zsol[1] = y[0][1];
					zsol[2] = y[0][2];
					dstsq = dell[0][0];
					return(dstsq);
				}
				/* check optimality of line segment 1-2 */
				d1[2] = dell[1][1] - dell[1][0];
				if (!(d1[2]<=0.0 || d2[2]<=0.0)) {
					sum = d1[2] + d2[2];
					als[0] = d1[2]/sum;
					als[1] = 1.0 - als[0];
					zsol[0] = y[1][0] + als[0]*(y[0][0] - y[1][0]);
					zsol[1] = y[1][1] + als[0]*(y[0][1] - y[1][1]);
					zsol[2] = y[1][2] + als[0]*(y[0][2] - y[1][2]);
					dstsq = zsol[0]*zsol[0] + zsol[1]*zsol[1] + zsol[2]*zsol[2];
					return(dstsq);
				}
				/* check optimality of vetex 2 */
				if (d1[2]<=0.0) {
					*nvs = 1;
					ris[0] = ris[1];
					rjs[0] = rjs[1];
					als[0] = d2[1];
					zsol[0] = y[1][0];
					zsol[1] = y[1][1];
					zsol[2] = y[1][2];
					dstsq = dell[1][1];
					y[0][0] = y[1][0];
					y[0][1] = y[1][1];
					y[0][2] = y[1][2];
					dell[0][0] = dell[1][1];
					return(dstsq);
				}
				break;
			} /* END case two points */
		case 3: 
			{	/* case of three points ... */
				/* check optimality of vertex 1 */
				d2[2] = dell[0][0] - dell[1][0];
				d3[4] = dell[0][0] - dell[2][0];
				if (!(d2[2]>0.0 || d3[4]>0.0)){
					*nvs = 1;
					als[0] = d1[0];
					zsol[0] = y[0][0];
					zsol[1] = y[0][1];
					zsol[2] = y[0][2];
					dstsq = dell[0][0];
					return(dstsq);
				}
				/* check optimality of line segment 1-2 */
				e132 = dell[1][0] - dell[2][1];
				d1[2] = dell[1][1] - dell[1][0];
				d3[6] = d1[2]*d3[4] + d2[2]*e132;
				if (!(d1[2]<=0.0 || d2[2]<= 0.0 || d3[6]>0.0)) {
					*nvs = 2;
					sum = d1[2] + d2[2];
					als[0] = d1[2]/sum;
					als[1] = 1.0 - als[0];
					zsol[0] = y[1][0] + als[0]*(y[0][0] - y[1][0]);
					zsol[1] = y[1][1] + als[0]*(y[0][1] - y[1][1]);
					zsol[2] = y[1][2] + als[0]*(y[0][2] - y[1][2]);
					dstsq = zsol[0]*zsol[0] + zsol[1]*zsol[1] + zsol[2]*zsol[2];
					return(dstsq);
				}
				/* check optimality of line segment 1-3 */
				e123 = dell[2][0] - dell[2][1];
				d1[4] = dell[2][2] - dell[2][0];
				d2[6] = d1[4]*d2[2] + d3[4]*e123;
				if (!(d1[4]<=0.0 || d2[6]>0.0 || d3[4]<=0.0)) {
					*nvs = 2;
					ris[1] = ris[2];
					rjs[1] = rjs[2];
					sum = d1[4] + d3[4];
					als[0] = d1[4]/sum;
					als[1] = 1.0 - als[0];
					zsol[0] = y[2][0] + als[0]*(y[0][0] - y[2][0]);
					zsol[1] = y[2][1] + als[0]*(y[0][1] - y[2][1]);
					zsol[2] = y[2][2] + als[0]*(y[0][2] - y[2][2]);
					dstsq = zsol[0]*zsol[0] + zsol[1]*zsol[1] + zsol[2]*zsol[2];
					y[1][0] = y[2][0];
					y[1][1] = y[2][1];
					y[1][2] = y[2][2];
					dell[1][0] = dell[2][0];
					dell[1][1] = dell[2][2];
					return(dstsq);
				}
				/* check optimality of face 123 */
				e213 = -e123;
				d2[5] = dell[2][2] - dell[2][1];
				d3[5] = dell[1][1] - dell[2][1];
				d1[6] = d2[5]*d1[2] + d3[5]*e213;
				if (!(d1[6]<=0.0 || d2[6]<=0.0 || d3[6]<=0.0)) {
					sum = d1[6] + d2[6] + d3[6];
					als[0] = d1[6]/sum;
					als[1] = d2[6]/sum;
					als[2] = 1.0 - als[0] - als[1];
					zsol[0] = y[2][0] + als[0]*(y[0][0] - y[2][0]) +
					    als[1]*(y[1][0] - y[2][0]);
					zsol[1] = y[2][1] + als[0]*(y[0][1] - y[2][1]) +
					    als[1]*(y[1][1] - y[2][1]);
					zsol[2] = y[2][2] + als[0]*(y[0][2] - y[2][2]) +
					    als[1]*(y[1][2] - y[2][2]);
					dstsq = zsol[0]*zsol[0] + zsol[1]*zsol[1] + zsol[2]*zsol[2];
					return(dstsq);
				}
				/* check optimality of vertex 2 */
				if (!(d1[2]>0.0 || d3[5]>0.0)) {
					*nvs = 1;
					ris[0] = ris[1];
					rjs[0] = rjs[1];
					als[0] = d2[1];
					zsol[0] = y[1][0];
					zsol[1] = y[1][1];
					zsol[2] = y[1][2];
					dstsq = dell[1][1];
					y[0][0] = y[1][0];
					y[0][1] = y[1][1];
					y[0][2] = y[1][2];
					dell[0][0] = dell[1][1];
					return(dstsq);
				}
				/* check optimality of vertex 3 */
				if (!(d1[4]>0.0 || d2[5]>0.0)) {
					*nvs = 1;
					ris[0] = ris[2];
					rjs[0] = rjs[2];
					als[0] = d3[3];
					zsol[0] = y[2][0];
					zsol[1] = y[2][1];
					zsol[2] = y[2][2];
					dstsq = dell[2][2];
					y[0][0] = y[2][0];
					y[0][1] = y[2][1];
					y[0][2] = y[2][2];
					dell[0][0] = dell[2][2];
					return(dstsq);
				}
				/* check optimality of line segment 2-3 */
				if (!(d1[6]>0.0 || d2[5]<=0.0 || d3[5]<=0.0)) {
					*nvs = 2;
					ris[0] = ris[2];
					rjs[0] = rjs[2];
					sum = d2[5] + d3[5];
					als[1] = d2[5]/sum;
					als[0] = 1.0 - als[1];
					zsol[0] = y[2][0] + als[1]*(y[1][0] - y[2][0]);
					zsol[1] = y[2][1] + als[1]*(y[1][1] - y[2][1]);
					zsol[2] = y[2][2] + als[1]*(y[1][2] - y[2][2]);
					dstsq = zsol[0]*zsol[0] + zsol[1]*zsol[1] + zsol[2]*zsol[2];
					y[0][0] = y[2][0];
					y[0][1] = y[2][1];
					y[0][2] = y[2][2];
					dell[1][0] = dell[2][1];
					dell[0][0] = dell[2][2];
					return(dstsq);
				}
				break;
			} /* END case three points */
		case 4:	
			{ /* case of four points ... */
				/* check optimality of vertex 1 */
				d2[2] = dell[0][0] - dell[1][0];
				d3[4] = dell[0][0] - dell[2][0];
				d4[8] = dell[0][0] - dell[3][0];
				if (!(d2[2]>0.0 || d3[4]>0.0 || d4[8]>0.0)) {
					*nvs = 1;
					als[0] = d1[0];
					zsol[0] = y[0][0];
					zsol[1] = y[0][1];
					zsol[2] = y[0][2];
					dstsq = dell[0][0];
					return(dstsq);
				}
				/* check optimality of line segment 1-2 */
				e132 = dell[1][0] - dell[2][1];
				e142 = dell[1][0] - dell[3][1];
				d1[2] = dell[1][1] - dell[1][0];
				d3[6] = d1[2]*d3[4] + d2[2]*e132;
				d4[11] = d1[2]*d4[8] + d2[2]*e142;
				if (!(d1[2]<=0.0 || d2[2]<=0.0 ||
				    d3[6]>0.0 || d4[11]>0.0)) {
					*nvs = 2;
					sum = d1[2] + d2[2];
					als[0] = d1[2]/sum;
					als[1] = 1.0 - als[0];
					zsol[0] = y[1][0] + als[0]*(y[0][0] - y[1][0]);
					zsol[1] = y[1][1] + als[0]*(y[0][1] - y[1][1]);
					zsol[2] = y[1][2] + als[0]*(y[0][2] - y[1][2]);
					dstsq = zsol[0]*zsol[0] + zsol[1]*zsol[1] + zsol[2]*zsol[2];
					return(dstsq);
				}
				/* check optimality of line segment 1-3 */
				e123 = dell[2][0] - dell[2][1];
				e143 = dell[2][0] - dell[3][2];
				d1[4] = dell[2][2] - dell[2][0];
				d2[6] = d1[4]*d2[2] + d3[4]*e123;
				d4[12] = d1[4]*d4[8] + d3[4]*e143;
				if (!(d1[4]<=0.0 || d2[6]>0.0 ||
				    d3[4]<=0.0 || d4[12]>0.0)) {
					*nvs = 2;
					ris[1] = ris[2];
					rjs[1] = rjs[2];
					sum = d1[4] + d3[4];
					als[0] = d1[4]/sum;
					als[1] = 1.0 - als[0];
					zsol[0] = y[2][0] + als[0]*(y[0][0] - y[2][0]);
					zsol[1] = y[2][1] + als[0]*(y[0][1] - y[2][1]);
					zsol[2] = y[2][2] + als[0]*(y[0][2] - y[2][2]);
					dstsq = zsol[0]*zsol[0] + zsol[1]*zsol[1] + zsol[2]*zsol[2];
					y[1][0] = y[2][0];
					y[1][1] = y[2][1];
					y[1][2] = y[2][2];
					dell[1][0] = dell[2][0];
					dell[1][1] = dell[2][2];
					return(dstsq);
				}
				/* check optimality of face 123 */
				d2[5] = dell[2][2] - dell[2][1];
				d3[5] = dell[1][1] - dell[2][1];
				e213 = -e123;
				d1[6] = d2[5]*d1[2] + d3[5]*e213;
				d4[14] = d1[6]*d4[8] + d2[6]*e142 + d3[6]*e143;
				if (!(d1[6]<=0.0 || d2[6]<=0.0 ||
				    d3[6]<=0.0 || d4[14]>0.0)) {
					*nvs = 3;
					sum = d1[6] + d2[6] + d3[6];
					als[0] = d1[6]/sum;
					als[1] = d2[6]/sum;
					als[2] = 1.0 - als[0] - als[1];
					zsol[0] = y[2][0] + als[0]*(y[0][0] - y[2][0]) +
					    als[1]*(y[1][0] - y[2][0]);
					zsol[1] = y[2][1] + als[0]*(y[0][1] - y[2][1]) +
					    als[1]*(y[1][1] - y[2][1]);
					zsol[2] = y[2][2] + als[0]*(y[0][2] - y[2][2]) +
					    als[1]*(y[1][2] - y[2][2]);
					dstsq = zsol[0]*zsol[0] + zsol[1]*zsol[1] + zsol[2]*zsol[2];
					return(dstsq);
				}
				/* check optimality of line segment 1-4 */
				e124 = dell[3][0] - dell[3][1];
				e134 = dell[3][0] - dell[3][2];
				d1[8] = dell[3][3] - dell[3][0];
				d2[11] = d1[8]*d2[2] + d4[8]*e124;
				d3[12] = d1[8]*d3[4] + d4[8]*e134;
				if (!(d1[8]<=0.0 || d2[11]>0.0 ||
				    d3[12]>0.0 || d4[8]<=0.0)) {
					*nvs = 2;
					ris[1] = ris[3];
					rjs[1] = rjs[3];
					sum = d1[8] + d4[8];
					als[0] = d1[8]/sum;
					als[1] = 1.0 - als[0];
					zsol[0] = y[3][0] + als[0]*(y[0][0] - y[3][0]);
					zsol[1] = y[3][1] + als[0]*(y[0][1] - y[3][1]);
					zsol[2] = y[3][2] + als[0]*(y[0][2] - y[3][2]);
					dstsq = zsol[0]*zsol[0] + zsol[1]*zsol[1] + zsol[2]*zsol[2];
					y[1][0] = y[3][0];
					y[1][1] = y[3][1];
					y[1][2] = y[3][2];
					dell[1][0] = dell[3][0];
					dell[1][1] = dell[3][3];
					return(dstsq);
				}
				/* check optimality of face 1-2-4 */
				d2[9] = dell[3][3] - dell[3][1];
				d4[9] = dell[1][1] - dell[3][1];
				e214 = -e124;
				d1[11] = d2[9]*d1[2] + d4[9]*e214;
				d3[14] = d1[11]*d3[4] + d2[11]*e132 + d4[11]*e134;
				if (!(d1[11]<=0.0 || d2[11]<=0.0 ||
				    d3[14]>0.0 || d4[11]<=0.0)) {
					*nvs = 3;
					ris[2] = ris[3];
					rjs[2] = rjs[3];
					sum = d1[11] + d2[11] + d4[11];
					als[0] = d1[11]/sum;
					als[1] = d2[11]/sum;
					als[2] = 1.0 - als[0] - als[1];
					zsol[0] = y[3][0] + als[0]*(y[0][0] - y[3][0]) +
					    als[1]*(y[1][0] - y[3][0]);
					zsol[1] = y[3][1] + als[0]*(y[0][1] - y[3][1]) +
					    als[1]*(y[1][1] - y[3][1]);
					zsol[2] = y[3][2] + als[0]*(y[0][2] - y[3][2]) +
					    als[1]*(y[1][2] - y[3][2]);
					dstsq = zsol[0]*zsol[0] + zsol[1]*zsol[1] + zsol[2]*zsol[2];
					y[2][0] = y[3][0];
					y[2][1] = y[3][1];
					y[2][2] = y[3][2];
					dell[2][0] = dell[3][0];
					dell[2][1] = dell[3][1];
					dell[2][2] = dell[3][3];
					return(dstsq);
				}
				/* check optimality of face 1-3-4 */
				d3[10] = dell[3][3] - dell[3][2];
				d4[10] = dell[2][2] - dell[3][2];
				e314 = -e134;
				d1[12] = d3[10]*d1[4] + d4[10]*e314;
				d2[14] = d1[12]*d2[2] + d3[12]*e123 + d4[12]*e124;
				if (!(d1[12]<=0.0 || d2[14]>0.0 ||
				    d3[12]<=0.0 || d4[12]<=0.0)) {
					*nvs = 3;
					ris[1] = ris[3];
					rjs[1] = rjs[3];
					sum = d1[12] + d3[12] + d4[12];
					als[0] = d1[12]/sum;
					als[2] = d3[12]/sum;
					als[1] = 1.0 - als[0] - als[2];
					zsol[0] = y[3][0] + als[0]*(y[0][0] - y[3][0]) +
					    als[2]*(y[2][0] - y[3][0]);
					zsol[1] = y[3][1] + als[0]*(y[0][1] - y[3][1]) +
					    als[2]*(y[2][1] - y[3][1]);
					zsol[2] = y[3][2] + als[0]*(y[0][2] - y[3][2]) +
					    als[2]*(y[2][2] - y[3][2]);
					dstsq = zsol[0]*zsol[0] + zsol[1]*zsol[1] + zsol[2]*zsol[2];
					y[1][0] = y[3][0];
					y[1][1] = y[3][1];
					y[1][2] = y[3][2];
					dell[1][0] = dell[3][0];
					dell[1][1] = dell[3][3];
					dell[2][1] = dell[3][2];
					return(dstsq);
				}
				/* check optimality of the hull of all 4 points */
				e243 = dell[2][1] - dell[3][2];
				d4[13] = d2[5]*d4[9] + d3[5]*e243;
				e234 = dell[3][1] - dell[3][2];
				d3[13] = d2[9]*d3[5] + d4[9]*e234;
				e324 = -e234;
				d2[13] = d3[10]*d2[5] + d4[10]*e324;
				d1[14] = d2[13]*d1[2] + d3[13]*e213 + d4[13]*e214;
				if (!(d1[14]<=0.0 || d2[14]<=0.0 ||
				    d3[14]<=0.0 || d4[14]<=0.0)) {
					sum = d1[14] + d2[14] + d3[14] + d4[14];
					als[0] = d1[14]/sum;
					als[1] = d2[14]/sum;
					als[2] = d3[14]/sum;
					als[3] = 1.0 - als[0] - als[1] - als[2];
					zsol[0] = als[0]*y[0][0] + als[1]*y[1][0] +
					    als[2]*y[2][0] + als[3]*y[3][0];
					zsol[1] = als[0]*y[0][1] + als[1]*y[1][1] +
					    als[2]*y[2][1] + als[3]*y[3][1];
					zsol[2] = als[0]*y[0][2] + als[1]*y[1][2] +
					    als[2]*y[2][2] + als[3]*y[3][2];
					dstsq = zsol[0]*zsol[0] + zsol[1]*zsol[1] + zsol[2]*zsol[2];
					return(dstsq);
				}
				/* check optimality of vertex 2 */
				if (!(d1[2]>0.0 || d3[5]>0.0 || d4[9]>0.0)) {
					*nvs = 1;
					ris[0] = ris[1];
					rjs[0] = rjs[1];
					als[0] = d2[1];
					zsol[0] = y[1][0];
					zsol[1] = y[1][1];
					zsol[2] = y[1][2];
					dstsq = dell[1][1];
					y[0][0] = y[1][0];
					y[0][1] = y[1][1];
					y[0][2] = y[1][2];
					dell[0][0] = dell[1][1];
					return(dstsq);
				}
				/* check optimality of vertex 3 */
				if (!(d1[4]>0.0 || d2[5]>0.0 || d4[10]>0.0)) {
					*nvs = 1;
					ris[0] = ris[2];
					rjs[0] = rjs[2];
					als[0] = d3[3];
					zsol[0] = y[2][0];
					zsol[1] = y[2][1];
					zsol[2] = y[2][2];
					dstsq = dell[2][2];
					y[0][0] = y[2][0];
					y[0][1] = y[2][1];
					y[0][2] = y[2][2];
					dell[0][0] = dell[2][2];
					return(dstsq);
				}
				/* check optimality of vertex 4 */
				if (!(d1[8]>0.0 || d2[9]>0.0 || d3[10]>0.0)) {
					*nvs = 1;
					ris[0] = ris[3];
					rjs[0] = rjs[3];
					als[0] = d4[7];
					zsol[0] = y[3][0];
					zsol[1] = y[3][1];
					zsol[2] = y[3][2];
					dstsq = dell[3][3];
					y[0][0] = y[3][0];
					y[0][1] = y[3][1];
					y[0][2] = y[3][2];
					dell[0][0] = dell[3][3];
					return(dstsq);
				}
				/* check optimality of line segment 2-3 */
				if (!(d1[6]>0.0 || d2[5]<=0.0 ||
				    d3[5]<=0.0 || d4[13]>0.0)) {
					*nvs = 2;
					ris[0] = ris[2];
					rjs[0] = rjs[2];
					sum = d2[5] + d3[5];
					als[1] = d2[5]/sum;
					als[0] = 1.0 - als[1];
					zsol[0] = y[2][0] + als[1]*(y[1][0] - y[2][0]);
					zsol[1] = y[2][1] + als[1]*(y[1][1] - y[2][1]);
					zsol[2] = y[2][2] + als[1]*(y[1][2] - y[2][2]);
					dstsq = zsol[0]*zsol[0] + zsol[1]*zsol[1] + zsol[2]*zsol[2];
					y[0][0] = y[2][0];
					y[0][1] = y[2][1];
					y[0][2] = y[2][2];
					dell[1][0] = dell[2][1];
					dell[0][0] = dell[2][2];
					return(dstsq);
				}
				/* check optimality of line segment 2-4 */
				if (!(d1[11]>0.0 || d2[9]<=0.0 ||
				    d3[13]>0.0 || d4[9]<=0.0)) {
					*nvs = 2;
					ris[0] = ris[3];
					rjs[0] = rjs[3];
					sum = d2[9] + d4[9];
					als[1] = d2[9]/sum;
					als[0] = 1.0 - als[1];
					zsol[0] = y[3][0] + als[1]*(y[1][0] - y[3][0]);
					zsol[1] = y[3][1] + als[1]*(y[1][1] - y[3][1]);
					zsol[2] = y[3][2] + als[1]*(y[1][2] - y[3][2]);
					dstsq = zsol[0]*zsol[0] + zsol[1]*zsol[1] + zsol[2]*zsol[2];
					y[0][0] = y[3][0];
					y[0][1] = y[3][1];
					y[0][2] = y[3][2];
					dell[1][0] = dell[3][1];
					dell[0][0] = dell[3][3];
					return(dstsq);
				}
				/* check optimality of line segment 3-4 */
				if (!(d1[12]>0.0 || d2[13]>0.0 ||
				    d3[10]<=0.0 || d4[10]<=0.0)) {
					*nvs = 2;
					ris[0] = ris[2];
					ris[1] = ris[3];
					rjs[0] = rjs[2];
					rjs[1] = rjs[3];
					sum = d3[10] + d4[10];
					als[0] = d3[10]/sum;
					als[1] = 1.0 - als[0];
					zsol[0] = y[3][0] + als[0]*(y[2][0] - y[3][0]);
					zsol[1] = y[3][1] + als[0]*(y[2][1] - y[3][1]);
					zsol[2] = y[3][2] + als[0]*(y[2][2] - y[3][2]);
					dstsq = zsol[0]*zsol[0] + zsol[1]*zsol[1] + zsol[2]*zsol[2];
					y[0][0] = y[2][0];
					y[0][1] = y[2][1];
					y[0][2] = y[2][2];
					y[1][0] = y[3][0];
					y[1][1] = y[3][1];
					y[1][2] = y[3][2];
					dell[0][0] = dell[2][2];
					dell[1][0] = dell[3][2];
					dell[1][1] = dell[3][3];
					return(dstsq);
				}
				/* check optimality of face 2-3-4 */
				if (!(d1[14]>0.0 || d2[13]<=0.0 ||
				    d3[13]<=0.0 || d4[13]<=0.0)) {
					*nvs = 3;
					ris[0] = ris[3];
					rjs[0] = rjs[3];
					sum = d2[13] + d3[13] + d4[13];
					als[1] = d2[13]/sum;
					als[2] = d3[13]/sum;
					als[0] = 1.0 - als[1] - als[2];
					zsol[0] = y[3][0] + als[1]*(y[1][0] - y[3][0]) +
					    als[2]*(y[2][0] - y[3][0]);
					zsol[1] = y[3][1] + als[1]*(y[1][1] - y[3][1]) +
					    als[2]*(y[2][1] - y[3][1]);
					zsol[2] = y[3][2] + als[1]*(y[1][2] - y[3][2]) +
					    als[2]*(y[2][2] - y[3][2]);
					dstsq = zsol[0]*zsol[0] + zsol[1]*zsol[1] + zsol[2]*zsol[2];
					y[0][0] = y[3][0];
					y[0][1] = y[3][1];
					y[0][2] = y[3][2];
					dell[0][0] = dell[3][3];
					dell[1][0] = dell[3][1];
					dell[2][0] = dell[3][2];
					return(dstsq);
				}
				break;
			} /* END case four points */
		default: {
				fprintf(stderr,"Invalid value for nvs %d given \n",*nvs);
				break;
			}
		} /* END switch */
	}
/*======================================================================
**  The  backup procedure  begins ...                                  
**======================================================================*/
	switch (*nvs) {
	case 1: 
		{ /* case of a single point ... */
			dstsq = dell[0][0];
			als[0] = d1[0];
			zsol[0] = y[0][0];
			zsol[1] = y[0][1];
			zsol[2] = y[0][2];
			*backup = 1;
			return(dstsq);
			break;
		} /* END case of a single point */
	case 2: 
		{ /* case of two points ... */
			if (*backup) {
				d2[2] = dell[0][0] - dell[1][0];
				d1[2] = dell[1][1] - dell[1][0];
			}
			/* check vertex 1 */
			dstsq = dell[0][0];
			nvsd = 1;
			als[0] = d1[0];
			zsol[0] = y[0][0];
			zsol[1] = y[0][1];
			zsol[2] = y[0][2];
			iord[0] = 0;
			/* check line segment 1-2 */
			if (!(d1[2]<=0.0 || d2[2]<=0.0)) {
				sum = d1[2] + d2[2];
				alsd[0] = d1[2]/sum;
				alsd[1] = 1.0 - alsd[0];
				zsold[0] = y[1][0] + alsd[0]*(y[0][0] - y[1][0]);
				zsold[1] = y[1][1] + alsd[0]*(y[0][1] - y[1][1]);
				zsold[2] = y[1][2] + alsd[0]*(y[0][2] - y[1][2]);
				dstsqd = zsold[0]*zsold[0] + zsold[1]*zsold[1] +
				    zsold[2]*zsold[2];
				if (dstsqd < dstsq) {
					dstsq = dstsqd;
					nvsd = 2;
					als[0] = alsd[0];
					als[1] = alsd[1];
					zsol[0] = zsold[0];
					zsol[1] = zsold[1];
					zsol[2] = zsold[2];
					iord[0] = 0;
					iord[1] = 1;
				}
			}
			/* check vertex 2 */
			if (dell[1][1] < dstsq) {
				dstsq = dell[1][1];
				nvsd = 1;
				als[0] = d2[1];
				zsol[0] = y[1][0];
				zsol[1] = y[1][1];
				zsol[2] = y[1][2];
				iord[0] = 1;
			}
			break;
		} /* END case two points */
	case 3: 
		{ /* case of three points */
			if (*backup) {
				d2[2] = dell[0][0] - dell[1][0];
				d3[4] = dell[0][0] - dell[2][0];
				e132 = dell[1][0] - dell[2][1];
				d1[2] = dell[1][1] - dell[1][0];
				d3[6] = d1[2]*d3[4] + d2[2]*e132;
				e123 = dell[2][0] - dell[2][1];
				d1[4] = dell[2][2] - dell[2][0];
				d2[6] = d1[4]*d2[2] + d3[4]*e123;
				e213 = -e123;
				d2[5] = dell[2][2] - dell[2][1];
				d3[5] = dell[1][1] - dell[2][1];
				d1[6] = d2[5]*d1[2] + d3[5]*e213;
			}
			/* check vertex 1 */
			dstsq = dell[0][0];
			nvsd = 1;
			als[0] = d1[0];
			zsol[0] = y[0][0];
			zsol[1] = y[0][1];
			zsol[2] = y[0][2];
			iord[0] = 0;
			/* check line segment 1-2 */
			if (! (d1[2]<=0.0 || d2[2]<= 0.0)) {
				sum = d1[2] + d2[2];
				alsd[0] = d1[2]/sum;
				alsd[1] = 1.0 - alsd[0];
				zsold[0] = y[1][0] + alsd[0]*(y[0][0] - y[1][0]);
				zsold[1] = y[1][1] + alsd[0]*(y[0][1] - y[1][1]);
				zsold[2] = y[1][2] + alsd[0]*(y[0][2] - y[1][2]);
				dstsqd = zsold[0]*zsold[0] + zsold[1]*zsold[1] +
				    zsold[2]*zsold[2];
				if (dstsqd < dstsq) {
					dstsq = dstsqd;
					nvsd = 2;
					als[0] = alsd[0];
					als[1] = alsd[1];
					zsol[0] = zsold[0];
					zsol[1] = zsold[1];
					zsol[2] = zsold[2];
					iord[0] = 0;
					iord[1] = 1;
				}
			}
			/* check line segment 1-3 */
			if (!(d1[4]<=0.0 || d3[4]<=0.0)) {
				sum = d1[4] + d3[4];
				alsd[0] = d1[4]/sum;
				alsd[1] = 1.0 - alsd[0];
				zsold[0] = y[2][0] + alsd[0]*(y[0][0] - y[2][0]);
				zsold[1] = y[2][1] + alsd[0]*(y[0][1] - y[2][1]);
				zsold[2] = y[2][2] + alsd[0]*(y[0][2] - y[2][2]);
				dstsqd = zsold[0]*zsold[0] + zsold[1]*zsold[1] +
				    zsold[2]*zsold[2];
				if (dstsqd < dstsq) {
					dstsq = dstsqd;
					nvsd = 2;
					als[0] = alsd[0];
					als[1] = alsd[1];
					zsol[0] = zsold[0];
					zsol[1] = zsold[1];
					zsol[2] = zsold[2];
					iord[0] = 0;
					iord[1] = 2;
				}
			}
			/* check face 1-2-3 */
			if (!(d1[6]<=0.0 || d2[6]<=0.0 ||
			    d3[6]<=0.0)) {
				sum = d1[6] + d2[6] + d3[6];
				alsd[0] = d1[6]/sum;
				alsd[1] = d2[6]/sum;
				alsd[2] = 1.0 - alsd[0] - alsd[1];
				zsold[0] = y[2][0] + alsd[0]*(y[0][0] - y[2][0]) +
				    alsd[1]*(y[1][0] - y[2][0]);
				zsold[1] = y[2][1] + alsd[0]*(y[0][1] - y[2][1]) +
				    alsd[1]*(y[1][1] - y[2][1]);
				zsold[2] = y[2][2] + alsd[0]*(y[0][2] - y[2][2]) +
				    alsd[1]*(y[1][2] - y[2][2]);
				dstsqd = zsold[0]*zsold[0] + zsold[1]*zsold[1] +
				    zsold[2]*zsold[2];
				if (dstsqd < dstsq) {
					dstsq = dstsqd;
					nvsd = 3;
					als[0] = alsd[0];
					als[1] = alsd[1];
					als[2] = alsd[2];
					zsol[0] = zsold[0];
					zsol[1] = zsold[1];
					zsol[2] = zsold[2];
					iord[0] = 0;
					iord[1] = 1;
					iord[2] = 2;
				}
			}
			/* check vertex 2 */
			if (dell[1][1] < dstsq) {
				nvsd = 1;
				dstsq = dell[1][1];
				als[0] = d2[1];
				zsol[0] = y[1][0];
				zsol[1] = y[1][1];
				zsol[2] = y[1][2];
				iord[0] = 1;
			}
			/* check vertex 3 */
			if (dell[2][2] < dstsq) {
				nvsd = 1;
				dstsq = dell[2][2];
				als[0] = d3[3];
				zsol[0] = y[2][0];
				zsol[1] = y[2][1];
				zsol[2] = y[2][2];
				iord[0] = 2;
			}
			/* check line segment 2-3 */
			if (!(d2[5]<=0.0 || d3[5]<=0.0)) {
				sum = d2[5] + d3[5];
				alsd[1] = d2[5]/sum;
				alsd[0] = 1.0 - alsd[1];
				zsold[0] = y[2][0] + alsd[1]*(y[1][0] - y[2][0]);
				zsold[1] = y[2][1] + alsd[1]*(y[1][1] - y[2][1]);
				zsold[2] = y[2][2] + alsd[1]*(y[1][2] - y[2][2]);
				dstsqd = zsold[0]*zsold[0] + zsold[1]*zsold[1] +
				    zsold[2]*zsold[2];
				if (dstsqd < dstsq) {
					dstsq = dstsqd;
					nvsd = 2;
					als[0] = alsd[0];
					als[1] = alsd[1];
					zsol[0] = zsold[0];
					zsol[1] = zsold[1];
					zsol[2] = zsold[2];
					iord[0] = 2;
					iord[1] = 1;
				}
			}
			break;
		} /* END case three points */
	case 4: 
		{ /* case of four points */
			if (*backup) {
				d2[2] = dell[0][0] - dell[1][0];
				d3[4] = dell[0][0] - dell[2][0];
				d4[8] = dell[0][0] - dell[3][0];
				e132 = dell[1][0] - dell[2][1];
				e142 = dell[1][0] - dell[3][1];
				d1[2] = dell[1][1] - dell[1][0];
				d3[6] = d1[2]*d3[4] + d2[2]*e132;
				d4[11] = d1[2]*d4[8] + d2[2]*e142;
				e123 = dell[2][0] - dell[2][1];
				e143 = dell[2][0] - dell[3][2];
				d1[4] = dell[2][2] - dell[2][0];
				d2[6] = d1[4]*d2[2] + d3[4]*e123;
				d4[12] = d1[4]*d4[8] + d3[4]*e143;
				d2[5] = dell[2][2] - dell[2][1];
				d3[5] = dell[1][1] - dell[2][1];
				e213 = -e123;
				d1[6] = d2[5]*d1[2] + d3[5]*e213;
				d4[14] = d1[6]*d4[8] + d2[6]*e142 + d3[6]*e143;
				e124 = dell[3][0] - dell[3][1];
				e134 = dell[3][0] - dell[3][2];
				d1[8] = dell[3][3] - dell[3][0];
				d2[11] = d1[8]*d2[2] + d4[8]*e124;
				d3[12] = d1[8]*d3[4] + d4[8]*e134;
				d2[9] = dell[3][3] - dell[3][1];
				d4[9] = dell[1][1] - dell[3][1];
				e214 = -e124;
				d1[11] = d2[9]*d1[2] + d4[9]*e214;
				d3[14] = d1[11]*d3[4] + d2[11]*e132 + d4[11]*e134;
				d3[10] = dell[3][3] - dell[3][2];
				d4[10] = dell[2][2] - dell[3][2];
				e314 = -e134;
				d1[12] = d3[10]*d1[4] + d4[10]*e314;
				d2[14] = d1[12]*d2[2] + d3[12]*e123 + d4[12]*e124;
				e243 = dell[2][1] - dell[3][2];
				d4[13] = d2[5]*d4[9] + d3[5]*e243;
				e234 = dell[3][1] - dell[3][2];
				d3[13] = d2[9]*d3[5] + d4[9]*e234;
				e324 = -e234;
				d2[13] = d3[10]*d2[5] + d4[10]*e324;
				d1[14] = d2[13]*d1[2] + d3[13]*e213 + d4[13]*e214;
			}
			/* check vertex 1 */
			dstsq = dell[0][0];
			nvsd = 1;
			als[0] = d1[0];
			zsol[0] = y[0][0];
			zsol[1] = y[0][1];
			zsol[2] = y[0][2];
			iord[0] = 0;
			/* check line segment 1-2 */
			if (!(d1[2]<=0.0 || d2[2]<=0.0)) {
				sum = d1[2] + d2[2];
				alsd[0] = d1[2]/sum;
				alsd[1] = 1.0 - alsd[0];
				zsold[0] = y[1][0] + alsd[0]*(y[0][0] - y[1][0]);
				zsold[1] = y[1][1] + alsd[0]*(y[0][1] - y[1][1]);
				zsold[2] = y[1][2] + alsd[0]*(y[0][2] - y[1][2]);
				dstsqd = zsold[0]*zsold[0] + zsold[1]*zsold[1] +
				    zsold[2]*zsold[2];
				if (dstsqd < dstsq) {
					dstsq = dstsqd;
					nvsd = 2;
					als[0] = alsd[0];
					als[1] = alsd[1];
					zsol[0] = zsold[0];
					zsol[1] = zsold[1];
					zsol[2] = zsold[2];
					iord[0] = 0;
					iord[1] = 1;
				}
			}
			/* check line segment 1-3 */
			if (!(d1[4]<=0.0 || d3[4]<=0.0)) {
				sum = d1[4] + d3[4];
				alsd[0] = d1[4]/sum;
				alsd[1] = 1.0 - alsd[0];
				zsold[0] = y[2][0] + alsd[0]*(y[0][0] - y[2][0]);
				zsold[1] = y[2][1] + alsd[0]*(y[0][1] - y[2][1]);
				zsold[2] = y[2][2] + alsd[0]*(y[0][2] - y[2][2]);
				dstsqd = zsold[0]*zsold[0] + zsold[1]*zsold[1] +
				    zsold[2]*zsold[2];
				if (dstsqd < dstsq) {
					dstsq = dstsqd;
					nvsd = 2;
					als[0] = alsd[0];
					als[1] = alsd[1];
					zsol[0] = zsold[0];
					zsol[1] = zsold[1];
					zsol[2] = zsold[2];
					iord[0] = 0;
					iord[1] = 2;
				}
			}
			/* check face 1-2-3 */
			if (!(d1[6]<=0.0 || d2[6]<=0.0 ||
			    d3[6]<=0.0)) {
				sum = d1[6] + d2[6] + d3[6];
				alsd[0] = d1[6]/sum;
				alsd[1] = d2[6]/sum;
				alsd[2] = 1.0 - alsd[0] - alsd[1];
				zsold[0] = y[2][0] + alsd[0]*(y[0][0] - y[2][0]) +
				    alsd[1]*(y[1][0] - y[2][0]);
				zsold[1] = y[2][1] + alsd[0]*(y[0][1] - y[2][1]) +
				    alsd[1]*(y[1][1] - y[2][1]);
				zsold[2] = y[2][2] + alsd[0]*(y[0][2] - y[2][2]) +
				    alsd[1]*(y[1][2] - y[2][2]);
				dstsqd = zsold[0]*zsold[0] + zsold[1]*zsold[1] +
				    zsold[2]*zsold[2];
				if (dstsqd < dstsq) {
					dstsq = dstsqd;
					nvsd = 3;
					als[0] = alsd[0];
					als[1] = alsd[1];
					als[2] = alsd[2];
					zsol[0] = zsold[0];
					zsol[1] = zsold[1];
					zsol[2] = zsold[2];
					iord[0] = 0;
					iord[1] = 1;
					iord[2] = 2;
				}
			}
			/* check line segment 1-4 */
			if (!(d1[8]<=0.0 || d4[8]<=0.0)) {
				sum = d1[8] + d4[8];
				alsd[0] = d1[8]/sum;
				alsd[1] = 1.0 - alsd[0];
				zsold[0] = y[3][0] + alsd[0]*(y[0][0] - y[3][0]);
				zsold[1] = y[3][1] + alsd[0]*(y[0][1] - y[3][1]);
				zsold[2] = y[3][2] + alsd[0]*(y[0][2] - y[3][2]);
				dstsqd = zsold[0]*zsold[0] + zsold[1]*zsold[1] +
				    zsold[2]*zsold[2];
				if (dstsqd < dstsq) {
					dstsq = dstsqd;
					nvsd = 2;
					als[0] = alsd[0];
					als[1] = alsd[1];
					zsol[0] = zsold[0];
					zsol[1] = zsold[1];
					zsol[2] = zsold[2];
					iord[0] = 0;
					iord[1] = 3;
				}
			}
			/* check face 1-2-4 */
			if (!(d1[11]<=0.0 || d2[11]<=0.0 ||
			    d4[11]<=0.0)) {
				sum = d1[11] + d2[11] + d4[11];
				alsd[0] = d1[11]/sum;
				alsd[1] = d2[11]/sum;
				alsd[2] = 1.0 - alsd[0] - alsd[1];
				zsold[0] = y[3][0] + alsd[0]*(y[0][0] - y[3][0]) +
				    alsd[1]*(y[1][0] - y[3][0]);
				zsold[1] = y[3][1] + alsd[0]*(y[0][1] - y[3][1]) +
				    alsd[1]*(y[1][1] - y[3][1]);
				zsold[2] = y[3][2] + alsd[0]*(y[0][2] - y[3][2]) +
				    alsd[1]*(y[1][2] - y[3][2]);
				dstsqd = zsold[0]*zsold[0] + zsold[1]*zsold[1] +
				    zsold[2]*zsold[2];
				if (dstsqd < dstsq) {
					dstsq = dstsqd;
					nvsd = 3;
					als[0] = alsd[0];
					als[1] = alsd[1];
					als[2] = alsd[2];
					zsol[0] = zsold[0];
					zsol[1] = zsold[1];
					zsol[2] = zsold[2];
					iord[0] = 0;
					iord[1] = 1;
					iord[2] = 3;
				}
			}
			/* check face 1-3-4 */
			if (!(d1[12]<=0.0 ||
			    d3[12]<=0.0 || d4[12]<=0.0)) {
				sum = d1[12] + d3[12] + d4[12];
				alsd[0] = d1[12]/sum;
				alsd[2] = d3[12]/sum;
				alsd[1] = 1.0 - alsd[0] - alsd[2];
				zsold[0] = y[3][0] + alsd[0]*(y[0][0] - y[3][0]) +
				    alsd[2]*(y[2][0] - y[3][0]);
				zsold[1] = y[3][1] + alsd[0]*(y[0][1] - y[3][1]) +
				    alsd[2]*(y[2][1] - y[3][1]);
				zsold[2] = y[3][2] + alsd[0]*(y[0][2] - y[3][2]) +
				    alsd[2]*(y[2][2] - y[3][2]);
				dstsqd = zsold[0]*zsold[0] + zsold[1]*zsold[1] +
				    zsold[2]*zsold[2];
				if (dstsqd < dstsq) {
					dstsq = dstsqd;
					nvsd = 3;
					als[0] = alsd[0];
					als[1] = alsd[1];
					als[2] = alsd[2];
					zsol[0] = zsold[0];
					zsol[1] = zsold[1];
					zsol[2] = zsold[2];
					iord[0] = 0;
					iord[1] = 3;
					iord[2] = 2;
				}
			}
			/* check the hull of all 4 points */
			if (!(d1[14]<=0.0 || d2[14]<=0.0 ||
			    d3[14]<=0.0 || d4[14]<=0.0)) {
				sum = d1[14] + d2[14] + d3[14] + d4[14];
				alsd[0] = d1[14]/sum;
				alsd[1] = d2[14]/sum;
				alsd[2] = d3[14]/sum;
				alsd[3] = 1.0 - alsd[0] - alsd[1] - alsd[2];
				zsold[0] = alsd[0]*y[0][0] + alsd[1]*y[1][0] +
				    alsd[2]*y[2][0] + alsd[3]*y[3][0];
				zsold[1] = alsd[0]*y[0][1] + alsd[1]*y[1][1] +
				    alsd[2]*y[2][1] + alsd[3]*y[3][1];
				zsold[2] = alsd[0]*y[0][2] + alsd[1]*y[1][2] +
				    alsd[2]*y[2][2] + alsd[3]*y[3][2];
				dstsqd = zsold[0]*zsold[0] + zsold[1]*zsold[1] +
				    zsold[2]*zsold[2];
				if (dstsqd < dstsq) {
					dstsq = dstsqd;
					nvsd = 4;
					als[0] = alsd[0];
					als[1] = alsd[1];
					als[2] = alsd[2];
					als[3] = alsd[3];
					zsol[0] = zsold[0];
					zsol[1] = zsold[1];
					zsol[2] = zsold[2];
					iord[0] = 0;
					iord[1] = 1;
					iord[2] = 2;
					iord[3] = 3;
				}
			}
			/* check vertex 2 */
			if (dell[1][1] < dstsq) {
				nvsd = 1;
				dstsq = dell[1][1];
				als[0] = d2[1];
				zsol[0] = y[1][0];
				zsol[1] = y[1][1];
				zsol[2] = y[1][2];
				iord[0] = 1;
			}
			/* check vertex 3 */
			if (dell[2][2] < dstsq) {
				nvsd = 1;
				dstsq = dell[2][2];
				als[0] = d3[3];
				zsol[0] = y[2][0];
				zsol[1] = y[2][1];
				zsol[2] = y[2][2];
				iord[0] = 2;
			}
			/* check vertex 4 */
			if (dell[3][3] < dstsq) {
				nvsd = 1;
				dstsq = dell[3][3];
				als[0] = d4[7];
				zsol[0] = y[3][0];
				zsol[1] = y[3][1];
				zsol[2] = y[3][2];
				iord[0] = 3;
			}
			/* check line segment 2-3 */
			if (!(d2[5]<=0.0 || d3[5]<=0.0)) {
				sum = d2[5] + d3[5];
				alsd[1] = d2[5]/sum;
				alsd[0] = 1.0 - alsd[1];
				zsold[0] = y[2][0] + alsd[1]*(y[1][0] - y[2][0]);
				zsold[1] = y[2][1] + alsd[1]*(y[1][1] - y[2][1]);
				zsold[2] = y[2][2] + alsd[1]*(y[1][2] - y[2][2]);
				dstsqd = zsold[0]*zsold[0] + zsold[1]*zsold[1] +
				    zsold[2]*zsold[2];
				if (dstsqd < dstsq) {
					dstsq = dstsqd;
					nvsd = 2;
					als[0] = alsd[0];
					als[1] = alsd[1];
					zsol[0] = zsold[0];
					zsol[1] = zsold[1];
					zsol[2] = zsold[2];
					iord[0] = 2;
					iord[1] = 1;
				}
			}
			/* check line segment 2-4 */
			if (!(d2[9]<=0.0 || d4[9]<=0.0)) {
				sum = d2[9] + d4[9];
				alsd[1] = d2[9]/sum;
				alsd[0] = 1.0 - alsd[1];
				zsold[0] = y[3][0] + alsd[1]*(y[1][0] - y[3][0]);
				zsold[1] = y[3][1] + alsd[1]*(y[1][1] - y[3][1]);
				zsold[2] = y[3][2] + alsd[1]*(y[1][2] - y[3][2]);
				dstsqd = zsold[0]*zsold[0] + zsold[1]*zsold[1] +
				    zsold[2]*zsold[2];
				if (dstsqd < dstsq) {
					dstsq = dstsqd;
					nvsd = 2;
					als[0] = alsd[0];
					als[1] = alsd[1];
					zsol[0] = zsold[0];
					zsol[1] = zsold[1];
					zsol[2] = zsold[2];
					iord[0] = 3;
					iord[1] = 1;
				}
			}
			/* check line segment 3-4 */
			if (!(d3[10]<=0.0 || d4[10]<=0.0)) {
				sum = d3[10] + d4[10];
				alsd[0] = d3[10]/sum;
				alsd[1] = 1.0 - alsd[0];
				zsold[0] = y[3][0] + alsd[0]*(y[2][0] - y[3][0]);
				zsold[1] = y[3][1] + alsd[0]*(y[2][1] - y[3][1]);
				zsold[2] = y[3][2] + alsd[0]*(y[2][2] - y[3][2]);
				dstsqd = zsold[0]*zsold[0] + zsold[1]*zsold[1] +
				    zsold[2]*zsold[2];
				if (dstsqd < dstsq) {
					dstsq = dstsqd;
					nvsd = 2;
					als[0] = alsd[0];
					als[1] = alsd[1];
					zsol[0] = zsold[0];
					zsol[1] = zsold[1];
					zsol[2] = zsold[2];
					iord[0] = 2;
					iord[1] = 3;
				}
			}
			/* check face 2-3-4 */
			if (!(d2[13]<=0.0 ||
			    d3[13]<=0.0 || d4[13]<=0.0)) {
				sum = d2[13] + d3[13] + d4[13];
				alsd[1] = d2[13]/sum;
				alsd[2] = d3[13]/sum;
				alsd[0] = 1.0 - alsd[1] - alsd[2];
				zsold[0] = y[3][0] + alsd[1]*(y[1][0] - y[3][0]) +
				    alsd[2]*(y[2][0] - y[3][0]);
				zsold[1] = y[3][1] + alsd[1]*(y[1][1] - y[3][1]) +
				    alsd[2]*(y[2][1] - y[3][1]);
				zsold[2] = y[3][2] + alsd[1]*(y[1][2] - y[3][2]) +
				    alsd[2]*(y[2][2] - y[3][2]);
				dstsqd = zsold[0]*zsold[0] + zsold[1]*zsold[1] +
				    zsold[2]*zsold[2];
				if (dstsqd < dstsq) {
					dstsq = dstsqd;
					nvsd = 3;
					als[0] = alsd[0];
					als[1] = alsd[1];
					als[2] = alsd[2];
					zsol[0] = zsold[0];
					zsol[1] = zsold[1];
					zsol[2] = zsold[2];
					iord[0] = 3;
					iord[1] = 1;
					iord[2] = 2;
				}
			}
			break;
		} /* END case of four points */
	} /* END switch */
	/* final reordering */
	for (k=0;k<*nvs;k++) {
		risd[k]=ris[k];
		rjsd[k]=rjs[k];
		yd[k][0] = y[k][0];
		yd[k][1] = y[k][1];
		yd[k][2] = y[k][2];
		for (l=0;l<k;l++) {
			delld[k][l]=dell[k][l];
		}
		delld[k][k]=dell[k][k];
	}

	*nvs = nvsd;
	for (k=0;k<*nvs;k++) {
		kk = iord[k];
		ris[k] = risd[kk];
		rjs[k] = rjsd[kk];
		y[k][0] = yd[kk][0];
		y[k][1] = yd[kk][1];
		y[k][2] = yd[kk][2];
		for (l=0;l<k;l++) {
			ll = iord[l];
			if (kk >= ll)
				dell[k][l] = delld[kk][ll];
			else
				dell[k][l] = delld[ll][kk];
		}
		dell[k][k] = delld[kk][kk];
	}
	*backup = 1;
	return(dstsq);
}

