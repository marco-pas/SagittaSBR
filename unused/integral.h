/*


THIS IS NOT USED !!! Was just for reference


This code needs a few things:

1) "rayPool" class

1.1) "rayPool.rayTubeArray" : gets ray tubes --> "ray"
1.2) "rayPool.rayCount_" : counts rays
1.3) "rayPool.rayArea_" : gets ray area

2) "ray" class (from "rayTubeArray")

2.1) "ray.refCount_" : total number of hits for that ray
2.2) "ray.dist_" : total travelled distance after "refCount_" hits
2.3) "ray.pol_" : polarizaton of the ray (should not change if we have PEC)
2.4) "ray.dir_" : direction after the (last) hit
2.5) "ray.pos_" : position of the (last) hit



*/


void PhysicalOpticsIntegral( const 	< T >& rayPool, const Observation< T >& obs, T& rcs )
	{
		using namespace std;
		using namespace std::complex_literals;

		T freq = obs.frequency_;
		T angFreq = 2 * pi * freq;
		T waveLen = c0 / freq;
		T waveNum = 2 * pi / waveLen;

		RayTube< T >* rayPtr = rayPool.rayTubeArray_.get(); // gets ray tubes
		U32 rayCount = rayPool.rayCount_; // counts rays
		T rayArea = rayPool.rayArea_; // gets ray area

		LUV::Vec3< T > obsDir = obs.direction_;  // direction of the ray
		LUV::Vec3< T > obsDirSph = LUV::CtsToSph( obsDir ); // spherical coord ?

		T phi = obsDirSph[ 1 ];  // phi spherical
		T the = obsDirSph[ 2 ];  // theta spherical

		T cp = cos( phi ); // cos phi
		T sp = sin( phi ); // sin phi
		T ct = cos( the ); // cos theta
		T st = sin( the ); // sin theta

		LUV::Vec3< T > dirX( 1.0, 0.0, 0.0 ); 			// x unit vector
		LUV::Vec3< T > dirY( 0.0, 1.0, 0.0 ); 			// y unit vector
		LUV::Vec3< T > dirZ( 0.0, 0.0, 1.0 ); 			// z unit vector
		LUV::Vec3< T > dirP( -sp, cp, 0.0 ); 			// direction in phi
		LUV::Vec3< T > dirT( cp * ct, sp * ct, -st ); 	// direction in theta
		
		// wave vector in spherical coordinates
		LUV::Vec3< T > vecK = waveNum * ( ( dirX * cp + dirY * sp ) * st + dirZ * ct ); 
		
		complex< T > AU = 0;
		complex< T > AR = 0;

		complex< T > i( 0.0, 1.0 );

		for( U32 idRay = 0; idRay < rayCount; ++idRay ) // for all rays
		{
			RayTube< T >& ray = rayPtr[ idRay ];
			if( ray.refCount_ > 0 ) // only if ray has a hit
			{
				// calculate accumulated ray shift
				T kr = waveNum * ray.dist_;
				//T reflectionCoef = pow( -1.0, ray.refCount_ );

				// calculate reflection coeff
				T reflectionCoef = pow( 1.0, ray.refCount_ ); // 1.0 is the PEC

				// E(r) = exp(j*kr) * Polarization_Vector * Reflection_Loss
				LUV::Vec3< complex< T > > apE = exp( i * kr ) * ray.pol_ * reflectionCoef;
				// H(r) = - dir x E(r)
				LUV::Vec3< complex< T > > apH = -LUV::Cross( apE, ray.dir_ );

				// BU = Contribution to one polarization (likely Theta/Vertical)
				complex< T > BU = LUV::Dot( -( LUV::Cross( apE, -dirP ) + LUV::Cross( apH, dirT ) ), ray.dir_ );
				// BR = Contribution to orthogonal polarization (likely Phi/Horizontal)
				complex< T > BR = LUV::Dot( -( LUV::Cross( apE, dirT ) + LUV::Cross( apH, dirP ) ), ray.dir_ );

				// actual integral
				complex< T > factor = complex< T >( 0.0, ( ( waveNum * rayArea ) / ( 4.0 * pi ) ) ) * exp( -i * LUV::Dot( vecK, ray.pos_ ) );

				// accumulate the integral
				AU += BU * factor;
				AR += BR * factor;
			}

		}

		//std::cout << "AU: " << AU.real() << " + i" << AU.imag() << std::endl;
		//std::cout << "AR: " << AR.real() << " + i" << AR.imag() << std::endl;

		// final RCS
		rcs = 4.0 * pi * ( pow( abs( AU ), 2 ) + pow( abs( AR ), 2 ) ); // * 4 * pi

	}


