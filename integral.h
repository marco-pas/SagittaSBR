/*
void PhysicalOpticsIntegral( const RayPool< T >& rayPool, const Observation< T >& obs, T& rcs )
	{
		using namespace std;
		using namespace std::complex_literals;

		T freq = obs.frequency_;
		T angFreq = 2 * pi * freq;
		T waveLen = c0 / freq;
		T waveNum = 2 * pi / waveLen;

		RayTube< T >* rayPtr = rayPool.rayTubeArray_.get();
		U32 rayCount = rayPool.rayCount_;
		T rayArea = rayPool.rayArea_;

		LUV::Vec3< T > obsDir = obs.direction_;
		LUV::Vec3< T > obsDirSph = LUV::CtsToSph( obsDir );

		T phi = obsDirSph[ 1 ];
		T the = obsDirSph[ 2 ];

		T cp = cos( phi );
		T sp = sin( phi );
		T ct = cos( the );
		T st = sin( the );

		LUV::Vec3< T > dirX( 1.0, 0.0, 0.0 );
		LUV::Vec3< T > dirY( 0.0, 1.0, 0.0 );
		LUV::Vec3< T > dirZ( 0.0, 0.0, 1.0 );
		LUV::Vec3< T > dirP( -sp, cp, 0.0 );
		LUV::Vec3< T > dirT( cp * ct, sp * ct, -st );

		LUV::Vec3< T > vecK = waveNum * ( ( dirX * cp + dirY * sp ) * st + dirZ * ct );
		
		complex< T > AU = 0;
		complex< T > AR = 0;

		complex< T > i( 0.0, 1.0 );

		for( U32 idRay = 0; idRay < rayCount; ++idRay )
		{
			RayTube< T >& ray = rayPtr[ idRay ];
			if( ray.refCount_ > 0 )
			{
				T kr = waveNum * ray.dist_;
				//T reflectionCoef = pow( -1.0, ray.refCount_ );
				T reflectionCoef = pow( 1.0, ray.refCount_ );

				LUV::Vec3< complex< T > > apE = exp( i * kr ) * ray.pol_ * reflectionCoef;
				LUV::Vec3< complex< T > > apH = -LUV::Cross( apE, ray.dir_ );

				complex< T > BU = LUV::Dot( -( LUV::Cross( apE, -dirP ) + LUV::Cross( apH, dirT ) ), ray.dir_ );
				complex< T > BR = LUV::Dot( -( LUV::Cross( apE, dirT ) + LUV::Cross( apH, dirP ) ), ray.dir_ );

				complex< T > factor = complex< T >( 0.0, ( ( waveNum * rayArea ) / ( 4.0 * pi ) ) ) * exp( -i * LUV::Dot( vecK, ray.pos_ ) );

				AU += BU * factor;
				AR += BR * factor;
			}

		}

		//std::cout << "AU: " << AU.real() << " + i" << AU.imag() << std::endl;
		//std::cout << "AR: " << AR.real() << " + i" << AR.imag() << std::endl;

		rcs = 4.0 * pi * ( pow( abs( AU ), 2 ) + pow( abs( AR ), 2 ) ); // * 4 * pi

	}

*/

// check what they do