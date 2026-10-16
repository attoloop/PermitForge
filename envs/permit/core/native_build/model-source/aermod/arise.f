      SUBROUTINE AFLUXES
C***********************************************************************
C     AFLUXES Module of the AMS/EPA Regulatory Model - AERMOD
C
C     PURPOSE:    Compute Buoyancy Flux and other parameters for Aircraft Sources
C                 (Area/Volume Sources)
C
C     PROGRAMMER: Gavendra Pandey, UNC-IE, Chapel Hill, NC, USA &
C                 Akula Venkatram, UC-Riverside, CA, USA &
C                 Sarav Arunachalam, UNC-IE, Chapel Hill, NC, USA.
C
C     MODIFIED:   Updated the algorithm based on Pandey et al (2026) (In Review).
C                 The equation numbers are based on aircraft plume rise white paper.
C                 January 29, 2026 by Gavendra Pandey, UNC-IE, Chapel Hill, NC, USA.
C
C     MODIFIED:   Removed the NENG parameters from the buoyancy flux calculation for turbine engines,
C                 assuming that all the aircrafts in an hour treating as a single engine
C
C                 April 24th 2026 by Gavendra Pandey, UNC-IE, Chapel Hill, NC, USA.
C
C     DATE:    April 01, 2023
C
C     INPUTS:  Met Variables:
C                  Ambient temperature (K), TA
C                  Wind Speed (m/s), UP
C              Engine Parameters:
C                  Fuel Burn Rate (kg/s), MFUEL
C                  Thrust (N), THRUST
C                  Aircraft Velocity (m/s), VAA
C                  Air Fuel Ratio, AFR
C                  By-Pass Ratio, BYPR
C                  Rated Power (kilo watt), RPWR
C              Constants:
C                  Acceleration due to Gravity (m/s2), G = 9.80616
C                  Ambient Pressure (N/m2), PAA = 1.013e05
C                  Gas Constant of Air (J/kgK), RAA = 287
C                  Specific Heat of Exhaust Gases (J/kgK), CPA = 1003
C                  Heating Value of Fuel (J/kg), HFUEL = 43e06
C                  Number of engines per aircraft, NENG = 2                       ! Added on January 26th 2026
C                  Power Setting, PWST = 7% (Idle), 30% (Approach),
C                                 PWST = 85% (Climb-out), 100% (Take-off)
C                  Power setting is based on the Wayson et al. (2009)              
C
C     OUTPUTS: 
C                  Exhaust Velocity, VE (m/s)
C                  Exhaust Temperature, TE (K)
C                  Heat Rejection, QEE (W or J/s or kg-m2/s3)
C                  Exhaust Density, RHOE (kg/m3)
C                  Initial Plume Radius, R00 (m)
C                  Buoyancy Flux Line Thermal, FB (m3/s2)
C
C     CALLED FROM:   ARISE
C
C     REFERENCE: Equations are from aircraft plume rise white paper.
C***********************************************************************

C     Variable Declarations
      USE MAIN1
      IMPLICIT NONE
      CHARACTER MODNAM*12

C     Global Variables:
C      DOUBLE PRECISION  :: MFUEL, VAA, THRUST, MAIR, AFR, BYPR, RPWR
C      DOUBLE PRECISION  :: VE, QEE, TE, RHOE, R00, FB

C     Local Variables:
      DOUBLE PRECISION  :: UEFFA, MFUELL, THRUSTT, EFU                              ! Added on January 26th 2025
      DOUBLE PRECISION  :: MAIR, PWST, RPWRW, ETAT                                            

C     Fixed VALUES: Defined in MAIN1/MODULES
C      DOUBLE PRECISION, PARAMETER  ::
C     &                     PAA = 1.013D+5, RAA = 287.058D0,
C     &                     HFUEL = 4.3D+7, CPA = 1003.0D0,
C     &                     G = 9.80616D0, ETAT = 0.6D0, NENG = 2.0D0               ! Added on January 26th 2025

C     Variable Initializations
      MODNAM = 'AFLUXES'

C     Set initial effective parameters
      UEFFA  = UP                                                                   ! Effective Ambient Wind Speed (m/s); January 26th 2025

CC     Calculate the fuel burn rate and thrust per engine of aircraft                ! Added on January 26th 2025
C      MFUELL = MFUEL / NENG                                                         ! Fuel Burn Rate per engine (kg/s) for turbine engines only
C      THRUSTT = THRUST / NENG

      MFUELL = MFUEL
      THRUSTT = THRUST

C     Initialize the plume rise parameters
      QEE  = 1.0D-10
      TE   = 1.0D-10
      RHOE = 1.0D-10
      R00  = 1.0D-10    

C     For Turbofan and Turbojet Engines
      IF(BYPR .GT. 0.0D0) THEN
C     Check for Positive Fuel burn rate and Thrust  
            IF (MFUELL .GT. 0.0D0 .AND. THRUSTT .GT. 0.0D0) THEN
                  MAIR = MFUELL * AFR * (1.D0 + BYPR)                               ! Total Mass Flow (Equation 2 of white paper)
                  EFU  = VAA + UEFFA                                                ! Effective Velocity of aircraft
                  VE   = EFU + (THRUSTT / MAIR)                                     ! Effective Exhaust Velocity (Equation 3 of white paper)
                  QEE  = MFUELL*HFUEL-THRUSTT*(VE+ EFU)/2.0D0                       ! Thermal Power/Heat Rejection (Equation 11 of the white paper)
                  TE   = QEE/(MAIR * CPA) + TA                                      ! Exhaust Temperature (Equation 10 of the white paper)
            END IF 
C     For Non-Turbofan/Turbojet Engines or Shaft-based Engines, 
C     Turboprop, Turboshaft, Piston, and Helicopter
C     PWST is based on Table 4 of Wayson et al.(2009)
      ELSE IF (BYPR .EQ. -999.D0) THEN
C     Check for Positive Fuel burn rate
            IF (MFUEL .GT. 0.0D0) THEN
                  IF (AFR .EQ. 106.D0) THEN
                        PWST = 0.07D0                                               ! Idle/Taxi 
                  ELSE IF (AFR .EQ. 83.D0) THEN
                        PWST = 0.30D0                                               ! Approach/Landing
                  ELSE IF (AFR .EQ. 51.D0) THEN 
                        PWST = 0.85D0                                               ! Climb-out
                  ELSE
                        PWST = 1.0D0                                                ! Take-Off
                  END IF       
C                 kilo-Watt to Watt Conversion for Rated Power
                  RPWRW = RPWR * 1000.0D0
                  MAIR = MFUEL * (1.D0 +  AFR)
                  ETAT = ( PWST * RPWRW ) / ( MFUEL * HFUEL )                       ! Thermal Efficiency (Equation 30 of Pandey et al; 2024 (AE))
                  QEE =  MFUEL * HFUEL * (1.D0 - ETAT)                              ! Thermal Power (Equation 31 of Pandey et al; 2024 (AE))
                  TE = TA + (((1.D0 - ETAT) * HFUEL) / (AFR * CPA))                 ! Exhaust Temperature (Equation 32 of Pandey et al; 2024 (AE))
                  VE   = UEFFA + VAA                                                ! Effective Exhaust Velocity same as ambient wind speed
            END IF
      END IF

      RHOE = PAA / (RAA * TE)                                                       ! Exhaust Density (Equation 26 or 33 of Pandey et al; 2024 (AE))
      R00  = SQRT(MAIR / (PI * VE * RHOE))                                          ! Initial exhaust plume radius (Equation 10 of white paper)

      IF (QEE .LE. 1.0D-10 .OR. RHOE .LE. 1.0D-10) THEN
            FB = 1.0D-10
      ELSE
            FB   = G * QEE / (PI * RHOE * CPA * TA * VE)                            ! Line thermal Buoyancy Flux (Equation 12 or 18 of white paper)
      END IF

C     Check the Aircraft Debug Option to print the debug file
      IF(ARCFTDEBUG) THEN

       WRITE(ARCFTDBG,*) '===========================================',
     & '=============================================================',
     & '========================'  
       WRITE (ARCFTDBG,*)'SOURCE TYPE:  ',SRCTYP(ISRC),
     &     '    SOURCE ID:  ',SRCID(ISRC)  ,'      SOURCE NO.:  ',ISRC


      WRITE (ARCFTDBG,6001) KURDAT
      WRITE (ARCFTDBG,6002) MFUEL, THRUST, VAA, AFR, BYPR, RPWR,
     &                      VE, QEE, TE, RHOE, R00, FB

 6001 FORMAT(' YR/MN/DY/HR:  ', I8,
     &  //,2X,'<----------------------------------- SOURCE INFORMATION',
     1  ' ----------------------------------->',/,
     2  '  MFUEL    THRUST     VAA     AFR     BYPR       RPWR',
     3  '      VE       QEE          TE     RHOE      R00      FB',/,
     5  ' (KG/S)     (N)      (M/S)    (#)     (#)        (kW)',
     6  '     (M/S)     (W)          (K)   (KG/M3)    (M)',
     7  '   (M3/S2)',/)

 6002 FORMAT(1X,F5.2,2X,F10.1,1X,F7.2,3X,F5.1,3X,F5.2,3X,F8.1,4X,
     &      F7.2,1X,E12.4,1X,F6.1,3X,F5.2,2X,F7.1,4X,F6.2)

      END IF

      RETURN
      END


      SUBROUTINE ADELTAH ( XARG )
C***********************************************************************
C     ADELTAH Module of the AMS/EPA Regulatory Model - AERMOD
C
C     PURPOSE: To calculate plume rise for Aircraft Sources
C
C     PROGRAMMER: Gavendra Pandey, UNC-IE, Chapel Hill, NC, USA &
C                 Akula Venkatram, UC-Riverside, CA, USA &
C                 Sarav Arunachalam, UNC-IE, Chapel Hill, NC, USA.
C
C     MODIFIED:  Updated the algorithm by using the PLUME_TRAJ subroutine
C                instead of ASBLRIS and ACBLPRD.
C                January 29, 2026 by Gavendra Pandey, 
C                UNC-IE, Chapel Hill, NC, USA.
C
C     DATE:    April 01, 2023
C
C     INPUTS:  The distance at which to make the computation, XARG
C
C     OUTPUTS: Distance-Dependent Plume Rise, DHP (m)
C
C     CALLED FROM:  VCALC, ACALC
C
CCC The framework of the plume rise calculations is adopted from PRISE.f CC
C***********************************************************************

C     Variable Declarations
      USE MAIN1
      IMPLICIT NONE

      CHARACTER MODNAM*12
      DOUBLE PRECISION :: XARG, BVZI2

C     Variable Initializations
      MODNAM = 'ADELTAH'

C     ------------------------------------------------------------------
C     Case 1: Stable Atmosphere (or Unstable but plume is above mixing height)
C     ------------------------------------------------------------------
      IF( (STABLE  .OR.  (UNSTAB  .AND.  (HS .GE. ZI))) ) THEN

C        Call the PLUME_TRAJ subroutine that computes the plume rise
         CALL PLUME_TRAJ ( XARG )

C        For urban stable boundary layers, limit plume rise to 1.25*ZI - HSP
         IF (URBSTAB) THEN
C ---       "New fomulation" for v15181 to account for "partial penetration" of plume
C           above the urban "stable" mixing height, similar to approach used for
C           convective conditions
            IF( HSP+DHP .GE. ZI )THEN
C ---          Stack height + plume rise is .GE. ZI; use pseudo-penetrated plume
C              approach for URBAN SBL cases

C              Compute the square of the Brunt-Vaisala frequency at ZI, BVZI2

               BVZI2 = (G / PTATZI) * 0.005D0

C              Compute the value of PsubS, Eq. 26b in the 2nd reference (modified for aircraft sources by removing velocity term)
               PSUBS = FB /(BVZI2*(ZI-HSP)*(ZI-HSP)*(ZI-HSP))

C              Compute the ratio of delta(Hsub_e)/delta(Hsub_h), HEDHH
C              (Eq. 25 in the 2nd ref.
C              NOTE: 17.576 = (2.6)**3 and 0.296296 is (2/3)**3
               HEDHH = (17.576D0 * PSUBS + 0.296296D0) ** THIRD

C              Check the value of HEDHH and compute the plume penetration, P
               IF( HEDHH .LT. (2.0D0*THIRD) )THEN
                  PPF = 0.0D0

               ELSE IF( HEDHH .GT. 2.0D0 )THEN
                  PPF = 1.0D0

               ELSE
                  PPF = 1.5D0 - (1.0D0 / HEDHH)

               END IF

C ---          Include calculation of penetrated plume rise and height
               IF( PPF .GT. 0.0D0 )THEN

C                 Compute the plume height for the penetrated source
C                 (See Eq. 8 in the reference for Source 3)
                  IF (PPF .EQ. 1.0D0) THEN
                     DHP = HEDHH * (ZI-HSP)
                  ELSE
                     DHP = 0.75D0 * (ZI-HSP) * HEDHH + 0.5D0*(ZI-HSP)
                  END IF

               ELSE
C ---             Use "original" DHP value
                  DHP = DHP

               END IF

            END IF
         END IF            

C        Debug Output
         IF(ARCFTDEBUG) THEN
C           Print DHP (Calculated Rise) and current Met conditions
            WRITE(ARCFTDBG,6003) XARG, DHP, DHP+HSP, UP, BVF, ZI
6003        FORMAT(/,5X,'ADELTAH Stable Calc:',
     &            ' Downwind Dist = ',F10.1,' M;',
     &            ' Plume Rise = ',F6.1,' M; Eff Plume Ht = ',F6.1,' M;',
     &            ' STACK UP = ',F5.2,' M/S; STACK BVF = ',F7.4, '1/s;',
     &            ' Mixing Ht = ',F6.1,' M')
         ENDIF

C     ------------------------------------------------------------------
C     Case 2: Unstable / Convective Atmosphere (HS < ZI)
C     ------------------------------------------------------------------
      ELSEIF( UNSTAB )THEN
C           1. Direct Plume Rise
            CALL PLUME_TRAJ ( XARG )

C           2. Indirect Source Plume Rise
C           Compute  plume rise for the indirect plume           --- CALL ACBLPRN
            CALL ACBLPRN ( XARG )

C           3. Penetrated Plume Rise
            IF( PPF .GT. 0.0D0 )THEN
C                 Compute plume rise for the penetrated plume    --- CALL ACBLPR3
                  CALL ACBLPR3
            ELSE
C                 No plume penetration - plume rise is zero for this source
                  DHP3 = 0.0D0
            ENDIF

C           Debug Output
            IF(ARCFTDEBUG) THEN
C                 Print DHP (Calculated Rise) and current Met conditions
                  WRITE(ARCFTDBG,6004) XARG, DHP1, UP, ZI
6004              FORMAT(/,5X,'ADELTAH Convective Calc:',
     &            ' Downwind Dist = ',F10.1,' M;',
     &            ' Direct Plume Rise = ',F6.1,' M;',
     &            ' UP = ',F5.2,' M/S;',
     &            ' ZI = ',F6.1,' M')
            ENDIF

      ENDIF

      RETURN
      END

      SUBROUTINE PLUME_TRAJ ( XARG )
C***********************************************************************
C     PLUME_TRAJ Module of the AMS/EPA Regulatory Model - AERMOD
C
C     PURPOSE:  To calculate plume rise for the stable boundary layer
C               or for the convective boundary layer for Aircraft Sources
C               It solves the governing equations using adaptive RK45 (DOPRI45)
C               Computes full plume trajectory but stores All final state variables
C
C               Step 1: Initializes the plume at the engine exhaust.
C               Step 2: Steps downwind until distance XARG is reached or stability or ZI.
C               Step 3: Stores all final state variables (Z, R, Q, F).
C
C     PROGRAMMER: Gavendra Pandey, UNC-IE, Chapel Hill, NC, USA &
C                 Akula Venkatram, UC-Riverside, CA, USA.
C
C     MODIFIED:   Fixed the initial step size as 0.1 m, also added a feature to write 
C                 an individual trajectory file for maximum 5 sources having height 
C                 greater than 2 m at each hour. 
C                 This will enhance the debugging capabilities.
C                 Changed the error tolerances.
C                 Called the similarity_wind subroutine that gives the wind speeed and stability 
C                 parameter at plume height.
C
C                 April 24th 2026 by Gavendra Pandey, UNC-IE, Chapel Hill, NC, USA.
C
C     DATE:    January 29, 2026
C
C
C     INPUTS:  Downwind distance, XARG
C              Initial Plume Radius, R00
C              Exhaust Velocity, VE
C              Exhaust Temperature, TE
C              Exhaust Density, RHOE
C              Aircraft Speed, VAA
C              Stack Height, HS
C              Buoyancy flux, FB
C              Downwind distance, XARG
C              Wind speed at stack top, UP
C
C     OUTPUTS: Plume Rise, DHP/DHP1 (m)
C              X_END_PLUME, R_END_PLUME, Z_END_PLUME, Q_END_PLUME, FB_END_PLUME
C              DHP (Plume Rise), DHP1 (Plume Rise)
C
C     CALLED FROM:  VCALC, ACALC
C
C             Optimization Update:
C             Now uses FSAL (First Same As Last) property for the RK45 solver.
C             It pre-calculates the initial slope (K_SLOPE) and passes it 
C             into the loop to save computational cost.
C
C   References:   Pandey et al (2026) In Review/White Paper
C
C***********************************************************************

C     Variable Declarations
      USE MAIN1
      IMPLICIT NONE

C --- Inputs
      DOUBLE PRECISION :: XARG                                                                  ! Downwind distance

C --- Local variables
      CHARACTER MODNAM*12
      DOUBLE PRECISION :: UEFFA, UEXIT
      DOUBLE PRECISION :: XSTART, HDELTA, Y_TRAJ(4)
      DOUBLE PRECISION :: ZP0, WP0, RMOM0
      DOUBLE PRECISION :: CONFT
      DOUBLE PRECISION :: X_END_PLUME, R_END_PLUME, Z_END_PLUME,
     &                    Q_END_PLUME, FB_END_PLUME

C     FSAL Optimization Variable: Stores the slope between steps
      DOUBLE PRECISION :: K_SLOPE(4)

C     RK45 (Runge-Kutta) tolerances, used locally
      DOUBLE PRECISION, PARAMETER  :: RELTOL = 1.0D-4, ABSTOL = 1.0D-6

C     Trajectory Debug File & Diagnostic Variables
      INTEGER, PARAMETER :: ITRAJ = 155                           
      CHARACTER(LEN=100) :: TRAJ_FNAME
      DOUBLE PRECISION   :: U_DIAG, N2_DIAG, VEL_PLM

C     Tracking for Multiple Sources and Execution Runs
C     Limit to 5000 sources if more than 5000 to check the criteria
      INTEGER, SAVE :: LAST_KURDAT(5000) ! Adjust size to max sources
      INTEGER, SAVE :: APPROVED_SOURCES(5) = (/ -1, -1, -1, -1, -1 /)
      INTEGER, SAVE :: MATCH_COUNT = 0
      LOGICAL, SAVE :: FIRST_RUN = .TRUE.
      LOGICAL       :: IS_APPROVED
      INTEGER       :: IDX

C     External Functions call here, only to use the velocity in the debug file
      DOUBLE PRECISION, EXTERNAL :: UT_FROM_CONSTRAINT

C     Fixed VALUES, these are laready defined in MAIN1
C      DOUBLE PRECISION, PARAMETER  ::
C     &                     PAA = 1.013D+5, RAA = 287.058D0,
C     &                     CPA = 1003.0D0, ALPHAM = 0.1D0, G = 9.80616D0, 
C     &                     NENG = 2.0D0, BETA1  = 0.6D0

C     Initialization: Only happens once when AERMOD starts for debug files
      IF (FIRST_RUN) THEN
            LAST_KURDAT = 0
            MATCH_COUNT = 0
            APPROVED_SOURCES = -1
            FIRST_RUN = .FALSE.
      END IF

C     Module name
      MODNAM = 'PLUME_TRAJ'

C     Set initial effective parameters
      UEFFA  = UP  

C     IDENTIFICATION LOGIC: Is this source one of our chosen 5?
      IS_APPROVED = .FALSE.
      
C     Check if current ISRC is already in our approved list
      DO IDX = 1, 5
         IF (APPROVED_SOURCES(IDX) .EQ. ISRC) IS_APPROVED = .TRUE.
      END DO

C     If not approved yet, see if it meets criteria and if we have space
      IF (.NOT. IS_APPROVED .AND. MATCH_COUNT .LT. 5) THEN
         IF (HS .EQ. 2.0D0 .AND. MFUEL .GT. 0.0D0) THEN
            MATCH_COUNT = MATCH_COUNT + 1
            APPROVED_SOURCES(MATCH_COUNT) = ISRC
            IS_APPROVED = .TRUE.
         END IF
      END IF

C     For turbofan and non-turbofan/shaft-based engines having greater
C     than 1.0D-10 buoyancy flux      
      IF ( FB .GT. 1.0D-10 ) THEN

C---------------------------------------------------------------
C       Check if Debug is on AND if this is one of the 5 sources
         IF (ARCFTDEBUG .AND. IS_APPROVED) THEN

            WRITE(TRAJ_FNAME, '(A, A, A, I8.8, A)') 
     &            'TRAJ_', TRIM(SRCID(ISRC)), '_', KURDAT, '.DAT'
            
C           Check if we are starting a NEW hour for this specific source
            IF (KURDAT .NE. LAST_KURDAT(ISRC)) THEN
C              First time in this hour: REPLACE wipes old files from previous runs
               OPEN(UNIT=ITRAJ, FILE=TRAJ_FNAME, STATUS='REPLACE')
               LAST_KURDAT(ISRC) = KURDAT
            ELSE
C              Second receptor or later in the SAME hour: APPEND
               OPEN(UNIT=ITRAJ, FILE=TRAJ_FNAME, STATUS='OLD', 
     &              POSITION='APPEND')
            END IF

C           Write standard AERMOD and Trajectory File Headers
            WRITE(ITRAJ, 7010) VERSN, TITLE1(1:68), RUNDAT, RUNTIM
            WRITE(ITRAJ, 7011) IREC, XARG, IDAY, IHOUR
            WRITE(ITRAJ, 7012) 'Dist', 'Radius', 'Height', 'Momentum', 
     &                         'Buoyancy', 'U_Amb', 'N2', 'Velocity',
     &                        'MixHeight',
     &                        'FuelBurnRate', 'Thrust', 'RatedPower',
     &                         'Speed', 'BYPR', 'AFR'
         END IF
C---------------------------------------------------------------

C           Compute the net exahaut velocity       
            UEXIT = VE - VAA

C------------------------------------------------------------------
C           1. Initialize plume state at engine exhaust
C------------------------------------------------------------------
C           Initial conditions at source/exhaust of the engine
            ZP0     = HSP           ! Initial Height (Stack/Release Height)
            WP0     = 0.0D0         ! Initial Vertical Velocity
            RMOM0   = R00           ! Initial Radius

C           Invariant from plume momentum (1): R002*UEXIT*(UEXIT - UEFFA) = CONFT
C           Enforce conservation of flux of momentum, couples plume radius
C           and vertical velocity through the plume rise trajectory
            CONFT = RMOM0**2.0D0 * UEXIT * (UEXIT - UEFFA)

C           State vector: Y = [RMOM0; ZP0; QP0; FB], QP0 = RMOM0^2 * WP0
            Y_TRAJ(1) = RMOM0                 ! Radius
            Y_TRAJ(2) = ZP0                   ! Height
            Y_TRAJ(3) = RMOM0**2.0D0 * WP0    ! Momentum Parameter
            Y_TRAJ(4) = FB                    ! Buoyancy Flux        

C           Initial step
            XSTART  = 0.0D0
C           Initial step size guess (Solver will adjust this automatically)
C            HDELTA = XARG / 100.0D0
            HDELTA = 0.1D0

C----------------- Debug Step 0 -------------------------------            
            IF (ARCFTDEBUG .AND. IS_APPROVED) THEN
               CALL SIMILARITY_WIND(Y_TRAJ(2), U_DIAG, N2_DIAG)
               VEL_PLM = UT_FROM_CONSTRAINT(Y_TRAJ(1), U_DIAG, CONFT)
               WRITE(ITRAJ, 7013) XSTART, Y_TRAJ(1), Y_TRAJ(2), 
     &                            Y_TRAJ(3), Y_TRAJ(4), U_DIAG, 
     &                            N2_DIAG, VEL_PLM, ZI, 
     &                            MFUEL, THRUST, RPWR,
     &                            VAA, BYPR, AFR
            END IF            
C---------------------------------------------------------------

C           ------------------------------------------------------------
C           2. Pre-Calculate Initial Slope (FSAL Optimization)
C           ------------------------------------------------------------
C           We calculate the slope at X=0 once. 
C           The solver will then update K_SLOPE internally for the next steps.
            CALL RHS_IN_X(Y_TRAJ, K_SLOPE, CONFT)

C---------------------------------------------------------------
C           3. Main RK45 integration loop
C---------------------------------------------------------------            
C           Loop until the current distance (XSTART) reaches the target (XARG)
100        CONTINUE

            IF (XSTART .LT. XARG) THEN                  

C                 1. CHECK: Have we hit the Mixing Height Ceiling?
                  IF (Y_TRAJ(2) .GE. ZI) THEN
C                       We hit the top of the boundary layer.
                        Y_TRAJ(2) = ZI      ! Clip exactly to ceiling
                        Y_TRAJ(3) = 0.0D0   ! Kill vertical momentum
C                       Force Buoyancy to Zero (simulating the cap killing the plume)
                        Y_TRAJ(4) = 0.0D0   ! Kill buoyancy
                        GOTO 200  ! Jump to end
                  END IF
C                 2. Check for equilibrium (Optional Efficiency Check) 
C                 (Has Buoyancy naturally run out? (Stable case))
C                 If Buoyancy is exhausted, we can stop early (avoid integrating flat line)
                  IF (Y_TRAJ(4) .LE. 1.0D-10) THEN
                        GOTO 200 
                  END IF

C                 Overshoot protection: Will the NEXT step cross mixing height (ZI)?
C                 If the current step size HDELTA would push us past ZI,
C                 we calculate the distance required to hit ZI exactly.
                  IF (UNSTAB .AND. (Y_TRAJ(2) + (Y_TRAJ(3)/
     &              MAX(Y_TRAJ(1)**2,1D-6))*HDELTA .GT. ZI)) THEN
                        ! Adjust step to stop right at the ceiling
                        ! This prevents the numerical "leak" past ZI
                        HDELTA = MAX(0.1D0, (ZI - Y_TRAJ(2)) / 
     &              MAX(Y_TRAJ(3)/MAX(Y_TRAJ(1)**2,1D-6), 1.0D-10))
                  END IF

C                 3. Take one adaptive step
C                 Pass K_SLOPE as the last argument
                  CALL DOPRI45_STEP(XSTART, Y_TRAJ, HDELTA, CONFT, 
     &                              RELTOL, ABSTOL, XARG, K_SLOPE)           

C-----------------For Debug Only ------------------------------- 
                  IF (ARCFTDEBUG .AND. IS_APPROVED) THEN
                     CALL SIMILARITY_WIND(Y_TRAJ(2), U_DIAG, N2_DIAG)
                     VEL_PLM = UT_FROM_CONSTRAINT(Y_TRAJ(1),
     &                                            U_DIAG,CONFT)
                     WRITE(ITRAJ, 7013) XSTART, Y_TRAJ(1), Y_TRAJ(2), 
     &                                  Y_TRAJ(3), Y_TRAJ(4), U_DIAG, 
     &                                  N2_DIAG, VEL_PLM, ZI,
     &                                  MFUEL, THRUST, RPWR,
     &                                  VAA, BYPR, AFR
                  END IF
C---------------------------------------------------------------
                  GOTO 100
            ENDIF

200         CONTINUE
            IF (ARCFTDEBUG .AND. IS_APPROVED) THEN
                  CLOSE(ITRAJ)
            END IF

C------------------------------------------------------------
C           4. Store final plume state
C------------------------------------------------------------
C           We have reached XARG. Unpack the final state from Y_TRAJ.

            X_END_PLUME  = XSTART       ! Final Distance
            R_END_PLUME  = Y_TRAJ(1)    ! Final Radius
            Z_END_PLUME  = Y_TRAJ(2)    ! Final Absolute Height
            Q_END_PLUME  = Y_TRAJ(3)    ! Final Momentum Parameter
            FB_END_PLUME = Y_TRAJ(4)    ! Final Buoyancy Flux            

C           ------------------------------------------------------------
C           5. Update standard plume rise parameters
C           ------------------------------------------------------------
C           Calculate Plume Rise (DHP)          
C            DHP = Z_END_PLUME                  ! Fixed this bug to account the actual plume height
            DHP = MAX(Z_END_PLUME - HSP, 0.0D0)

C           Calculate Plume Spread (PLUME_SPREAD) for use in dispersion
            PLUME_SPREAD = R_END_PLUME

C           ------------------------------------------------------------
C           6. Limit the plume rise to mixing height
C           ------------------------------------------------------------
C           Limit the plume rise to maximum
            DHP = MIN(DHP, ABS(ZI - HSP))
            DHP1 = DHP

      ELSE
            DHP = 0.0D0
            DHP1 = 0.0D0
            PLUME_SPREAD = 0.0D0
      END IF

C     FORMAT STATEMENTS for Debug Option
 7010 FORMAT('*** AERMOD - VERSION ',A6,' ***',52X,'*** ',A28,
     &       ' ***',2X,A8,/115X,' ***',2X,A8/)

 7011 FORMAT('# Receptor =', I6, '   Target_X =', F10.2, 
     &       '    Day =', I4, '    Hour =', I3)
 
 7012 FORMAT(15A15)
 
 7013 FORMAT(F15.2, F15.3, F15.2, E15.5, E15.5, F15.3, E15.5, F15.4,
     &          F15.2, E15.5, E15.5, F15.1, F15.2, F15.2, F15.1)
      RETURN
      END

      SUBROUTINE ACBLPRN ( XARG )
C***********************************************************************
C     ACBLPRN Module of the AMS/EPA Regulatory Model - AERMOD
C
C     PURPOSE: To calculate plume rise for the indirect plume in the
C              convective boundary layer for Aircraft Sources
C
C     PROGRAMMER: Gavendra Pandey, UNC-IE, Chapel Hill, NC, USA &
C                 Akula Venkatram, UC-Riverside, CA, USA.
C
C     DATE:     January 29, 2026
C
C     CHANGES:  Updated the algorithm based on Pandey et al (2026) (In Review).
C
C     CHANGES:  Equation for DHP2 revised per memorandum from Jeff Weil 
C               to AERMIC dated June 20, 1994.
C
C               Added XARG as formal argument for new formulation.
C               Roger Brode, PES, Inc. - 12/5/94
C     INPUTS:
C
C     OUTPUTS:  Plume Rise, DHP2 (m)
C
C     CALLED FROM:  ADELTAH
C
C     Assumptions:
C
C     References:   "A Dispersion Model for the Convective Boundary Layer",
C                 J. Weil, 8/17/93
C
CCC The framework of the plume rise calculations is adopted from PRISE.f CC
C***********************************************************************

C     Variable Declarations
      USE MAIN1
      IMPLICIT NONE
      CHARACTER MODNAM*12
      DOUBLE PRECISION :: XARG, RSUBH, RYRZ, DELHI

C     Variable Initializations
      MODNAM = 'ACBLPRN'

      RSUBH = BETA2 * (ZI - HSP)
      RYRZ  = RSUBH * RSUBH + 0.25D0*ASUBE*(LAMDAY**1.5D0) *
     &                        WSTAR*WSTAR*XARG*XARG/(UP*UP)
      DELHI = DSQRT( (2.D0*FB*ZI)/(ALPHAR*RYRZ) ) * (XARG/UP)
      DHP2  = DELHI

      RETURN
      END

      SUBROUTINE ACBLPR3
C***********************************************************************
C     ACBLPR3 Module of the AMS/EPA Regulatory Model - AERMOD
C
C     PURPOSE: To calculate plume rise for the penetrated plume in the
C              convective boundary layer for aircraft sources
C
C     PROGRAMMER: Jim Paumier and Roger Brode, PES, Inc
C
C     DATE:    September 30, 1993
C
C     INPUTS:  The ratio of delta(Hsub_e) to delta(Hsub_h), HEDHH
C              Mixing height, ZI
C              Source release height, HS
C
C     OUTPUTS: Plume Rise, DHP3 (m)
C
C     CALLED FROM: ADELTAH
C
C     Assumptions:
C
C     References:  "A Dispersion Model for the Convective Boundary
C                  Layer", J. Weil, 8/17/93
C                 "Plume Penetration of the CBL and Source 3: Source
C                  Strength and Plume Rise", J. Weil, 9/1/93
C
CCC The framework of the plume rise calculations is adopted from PRISE.f CC
C***********************************************************************

C     Variable Declarations
      USE MAIN1
      IMPLICIT NONE
      CHARACTER MODNAM*12

C     Variable Initializations
      MODNAM = 'ACBLPR3' ! Bug fixed from CBLPR3 to ACBLPR3

C     The plume rise for the penetrated source is delta(Hsub_3), given
C     by Eq. 9 in Jeff Weil's 9/1/93 document.  The variable HEDHH is
C     delta(Hsub_e)/delta(Hsub_h), calculated from Eq. 26a of Jeff Weil's
C     8/17/93 document, where delta(Hsub_h) is ZI-HS.

      IF (PPF .EQ. 1.0D0) THEN
         DHP3 = HEDHH * (ZI-HSP)
      ELSE
         DHP3 = 0.75D0 * (ZI-HSP) * HEDHH + 0.5D0 * (ZI-HSP)
      END IF

      RETURN
      END


      SUBROUTINE DOPRI45_STEP(X_DIST, Y_STATC, H_STEP, CONFTT,
     &                        RELTOL, ABSTOL, XENDC, K_SLOPE)  
C***********************************************************************
C     DOPRI45_STEP Module of the AMS/EPA Regulatory Model - AERMOD
C
C     PURPOSE: Performs one adaptive Runge–Kutta–Dormand–Prince (4th/5th order) step
C              for integrating a system of 4 ODEs with respect to downwind distance X.
C
C              Optimized Version (FSAL - First Same As Last):
C              This version expects the slope at the current X to be passed in
C              via K_SLOPE. It returns the slope at the new X in the same variable.
C              This avoids recalculating the derivative at the start of every step.
C
C     PROGRAMMER: Gavendra Pandey, UNC-IE, Chapel Hill, NC, USA &
C                 Akula Venkatram, UC-Riverside, CA, USA
C
C     MODIFIED:   Update at each step for better readability and remove several DO loops
C                 from earlier to minimize the computation time. 
C                 Also added the following references of this emipirical solver below.
C
C                 1. Dormand, J. R., & Prince, P. J. (1980). A family of embedded Runge-Kutta formulae. 
C                    Journal of Computational and Applied Mathematics, 6(1), 19-26. 
C                    https://doi.org/10.1016/0771-050X(80)90013-3
C                 2. Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007). 
C                    Numerical Recipes: The Art of Scientific Computing (3rd ed.). Cambridge University Press.
C                    (Specifically Section 16.2: Adaptive Stepsize Control for Runge-Kutta)
C
C                 April 24th 2026 by Gavendra Pandey, UNC-IE, Chapel Hill, NC, USA.
C
C     DATE:    January 29, 2026
C
C     INPUTS:  
C           X_DIST      : Current independent variable (updated on exit/output; e.g., Distance)
C           Y_STATC(4)  : State vector containing the variables we are solving for [RM, ZP, QP, FB] (updated on exit/output)
C           H_STEP      : Step size (adaptively updated)
C           CONFTT      : Constraint constant for velocity closure (used in UT calculation)
C           ALPHAM      : Entrainment Constant (Momentum Part of ARISE) (fixed used from main1 in RHS_IN_X)
C           BETA1       : Coeff. in the calculation of 'direct' plume rise (fixed used from main1 in RHS_IN_X)
C           RELTOL      : Relative error tolerance (accuracy requirement)
C           ABSTOL      : Absolute error tolerance (accuracy requirement)
C           XENDC       : Downwind distance, XARG (the final target value of X)
C           K_SLOPE(4)  : [IN] Slope at the START of the step (equivalent to K1)
C           Y_TEMP(4)   : To store the local data for further use in next step
C
C     OUTPUTS: 
C           X_DIST	: Updated to X_DIST + H_STEP (if successful)
C           Y_STATC(4)  : Updated to new state vector (if successful)
C           H_STEP      : Updated with the optimal step size for the NEXT step
C           K_SLOPE(4)  : [OUT] Slope at the END of the step (becomes K1 for next step)
C
C     CALLED FROM:  PLUME_TRAJ
C
C     References:   Dormand–Prince RK45 with local error control
C***********************************************************************
C     Variable Declarations
      IMPLICIT NONE

C     Arguments/Local Variables
      DOUBLE PRECISION X_DIST, Y_STATC(4), H_STEP, CONFTT
      DOUBLE PRECISION RELTOL, ABSTOL, XENDC
      DOUBLE PRECISION K_SLOPE(4)

C     Internal/Local variables for Runge-Kutta stages (slopes)
C     Note: K1 is not declared here; we use K_SLOPE instead.
      DOUBLE PRECISION K2(4), K3(4), K4(4)
      DOUBLE PRECISION K5(4), K6(4), K7(4)

C     Solutions: Y5 is 5th order estimate, Y4 is 4th order estimate
      DOUBLE PRECISION Y4(4), Y5(4), Y_TEMP(4)

C     Error estimation variables
      DOUBLE PRECISION ERR, SCALE, ERR_SAFE
      INTEGER II

C     Step size control constants
      DOUBLE PRECISION SAFETY, MINFAC, MAXFAC
      PARAMETER (SAFETY = 0.9D0,   ! Safety margin (90% of max theoretical step)
     &           MINFAC = 0.2D0,   ! Don't shrink step by more than 5x
     &           MAXFAC = 5.0D0)   ! Don't grow step by more than 5x

C     ------------------------------------------------------------------
C     Start of Step Calculation
C     ------------------------------------------------------------------
C     Label 101: Retry point. If the step is rejected, we jump back here
C     with a smaller H_STEP. We do NOT need to recalculate K_SLOPE (K1).
101   CONTINUE

C     Stage 2: Calculate slope at Y_STATC + H_STEP*(1/5) using slope K1 (K_SLOPE)
      Y_TEMP = Y_STATC + H_STEP * (1.0D0/5.0D0) * K_SLOPE
      CALL RHS_IN_X(Y_TEMP, K2, CONFTT)

C     Stage 3: Calculate slope using K1 (K_SLOPE) and K2
C              X + H*(3/40 + 9/40) = X + H*(3/10)
      Y_TEMP = Y_STATC + H_STEP*(3.D0/40.D0*K_SLOPE + 9.D0/40.D0*K2)
      CALL RHS_IN_X(Y_TEMP, K3, CONFTT)

C     Stage 4: Calculate slope using K1 (K_SLOPE), K2, and K3
C              X + H*(44/45 - 56/15 + 32/9) = X + H*(4/5)
      Y_TEMP = Y_STATC + H_STEP*(44.D0/45.D0*K_SLOPE - 
     &         56.D0/15.D0*K2 + 32.D0/9.D0*K3)
      CALL RHS_IN_X(Y_TEMP, K4, CONFTT)

C     Stage 5: Standard Dopri45 coefficients
C              X + H*(approx 0.9)
      Y_TEMP = Y_STATC + H_STEP*(19372.D0/6561.D0*K_SLOPE - 
     &         25360.D0/2187.D0*K2 + 64448.D0/6561.D0*K3 - 
     &         212.D0/729.D0*K4)
      CALL RHS_IN_X(Y_TEMP, K5, CONFTT)

C     Stage 6: Standard Dopri45 coefficients
C              X + H
      Y_TEMP = Y_STATC + H_STEP*(9017.D0/3168.D0*K_SLOPE - 
     &         355.D0/33.D0*K2 + 46732.D0/5247.D0*K3 + 
     &         49.D0/176.D0*K4 - 5103.D0/18656.D0*K5)
      CALL RHS_IN_X(Y_TEMP, K6, CONFTT)

C----------------------------------------------------------------------
C     Calculate 5th Order Solution (Y5) - The "Candidate"
C----------------------------------------------------------------------
      Y5 = Y_STATC + H_STEP*(35.D0/384.D0*K_SLOPE
     &        + 500.D0/1113.D0*K3
     &        + 125.D0/192.D0*K4
     &        - 2187.D0/6784.D0*K5
     &        + 11.D0/84.D0*K6)

C     Stage 7: (K7) - FSAL Property
C     Calculate slope at the END of the interval (at Y5).
C     This is needed for the 4th order error estimate AND 
C     it becomes K1 for the NEXT step.
      CALL RHS_IN_X(Y5, K7, CONFTT)

C----------------------------------------------------------------------
C     Calculate 4th Order Solution (Y4) - The "Reference"
C----------------------------------------------------------------------
      Y4 = Y_STATC + H_STEP*(5179.D0/57600.D0*K_SLOPE
     &            + 0.D0 * K2               
     &            + 7571.D0/16695.D0*K3
     &            + 393.D0/640.D0*K4
     &            - 92097.D0/339200.D0*K5
     &            + 187.D0/2100.D0*K6
     &            + 1.D0/40.D0*K7)

C----------------------------------------------------------------------
C  Error estimation (difference between 4th and 5th order solutions)
C----------------------------------------------------------------------
      ERR = 0.D0
      DO II = 1,4
C           Scale tolerance based on the magnitude of the solution
            SCALE = ABSTOL + RELTOL*
     &           MAX(DABS(Y_STATC(II)),DABS(Y5(II)))
C           Prevent division by zero if state is exactly zero
            IF (SCALE .EQ. 0.D0) SCALE = 1.D-30
            
C           Normalized error: Difference between orders / tolerance
            ERR = MAX(ERR, DABS(Y5(II)-Y4(II))/SCALE)
      END DO

C     Prevent division by zero if error is exactly zero (perfect step)
      ERR_SAFE = MAX(ERR, 1.0D-30)

C----------------------------------------------------------------------
C  Step Size Control: Reject step if error too large - reduce step size and retry
C----------------------------------------------------------------------
      IF (ERR_SAFE .GT. 1.D0) THEN
            ! Step failed (Error > Tolerance). 
            ! Shrink H_STEP: New H = H * Safety * (1/Error)^0.25
            H_STEP = H_STEP * MAX(MINFAC, SAFETY*ERR_SAFE**(-0.25D0))
            
            ! Retry the step with the smaller H
            GOTO 101
      ENDIF

C----------------------------------------------------------------------
C  If we reached here, the step was successful.
C  Accept step: update solution and independent variable
C----------------------------------------------------------------------
C     1. Advance Independent Variable
      X_DIST = X_DIST + H_STEP
      
C     2. Update State Vector (Y_STATE = Y5)
      Y_STATC = Y5
            
      ! FSAL OPTIMIZATION: 
      ! Update K_SLOPE (K1) for the *next* call using K7.
      K_SLOPE = K7

C     ------------------------------------------------------------------
C     Calculate Step Size for NEXT Iteration
C     ------------------------------------------------------------------
C     1. Calculate ideal growth: New H = H * Safety * (1/Error)^0.2
      H_STEP = H_STEP * MIN(MAXFAC, SAFETY*ERR_SAFE**(-0.2D0))

C     2. OVERSHOOT PROTECTION:
C        Ensure the NEW step size does not push us past XENDC.
C        (Crucial: This must happen AFTER the growth calculation above)
      IF (X_DIST + H_STEP .GT. XENDC) THEN
            H_STEP = XENDC - X_DIST
      END IF

      RETURN
      END

      SUBROUTINE RHS_IN_X(Y_STATE, DYDX_OUT, CFT)
C***********************************************************************
C     RHS_IN_X Module of the AMS/EPA Regulatory Model - AERMOD
C
C     PURPOSE: Calculates the derivatives of the state vector Y_STATE with respect to 
C              downwind distance X for an integral plume model.
C
C              It models a "Jet in Crossflow" with entrainment and buoyancy effects.
C              It calculates how a plume grow in radius (RMOM00), rise in height (ZP00),
C              change (gains/loses) in momentum (QP00), and change in buoyancy flux (FBB)
C              by integrating a system of 4 ODEs with respect to downwind distance (X).
C
C              Note on DYDX_OUT(4): The buoyancy equation is written in a coupled form:
C              it depends on DYDX_OUT(1) (radius growth).
C              This assumes that the mixing which causes the plume to grow is the 
C              same mechanism mixing the stratified ambient air into the plume, reducing its buoyancy.
C
C   PROGRAMMER: Gavendra Pandey, UNC-IE, Chapel Hill, NC, USA &
C               Akula Venkatram, UC-Riverside, CA, USA
C
C   MODIFIED:   Limit all the variable here with some small values and also commented the calculation of 
C               wind speed and squared brunt vaisala frequency, instead of calculation here, 
C               call the newly added subroutine similarity_wind that has same logic as used earlier,
C               only benefit is now we can debug it at each step using trajectory file
C               April 24th 2026 by Gavendra Pandey, UNC-IE, Chapel Hill, NC, USA.
C
C   DATE:    January 29, 2026
C
C   INPUTS:  
C           Y_STATE(4)   : Current state vector [Radius, Height, Momentum, Buoyancy]
C           CFT          : Velocity Constraint
C           BVF          : Brunt-Vaisala Frequency (Stability) (DSQRT(G*TGP/PTP))
C           Constants:
C                 ALPHAM : Jet entrainment coefficient (shear entrainment)
C                 BETA1  : Plume entrainment coefficient (pressure/buoyancy entrainment)
C
C   OUTPUTS: DYDX_OUT(4): The calculated slopes (derivatives) used by the ODE solver.
C
C   CALLED FROM:  DOPRI45_STEP
C
C***********************************************************************
C     Module Imports
      USE MAIN1
      IMPLICIT NONE

C     Input/Output Variables
      DOUBLE PRECISION Y_STATE(4), DYDX_OUT(4)
      DOUBLE PRECISION CFT

C     Functions Called
      DOUBLE PRECISION UT_FROM_CONSTRAINT  
      EXTERNAL UT_FROM_CONSTRAINT          ! OPTIONAL: Good practice to declare external functions      

C     Fixed VALUES/Constants, these are already defined in MAIN1
C      DOUBLE PRECISION, PARAMETER  ::
C     &                     ALPHAM = 0.1D0, BETA1 = 0.6D0

C     Local Variables
      DOUBLE PRECISION RMOM00, ZP00       ! State aliases for readability: Radius and Height
      DOUBLE PRECISION QP00, FBB          ! State aliases for readability: Momentum and Buoyancy
      DOUBLE PRECISION UEFFANN            ! Ambient wind speed at plume height, Z
      DOUBLE PRECISION UTT                ! Total effective plume velocity
      DOUBLE PRECISION WPPP               ! Vertical plume velocity (w)
      DOUBLE PRECISION N22                ! Brunt-Vaisala frequency squared (Real Stability)
      DOUBLE PRECISION N22_DECAY          ! Stability used for Buoyancy Decay (Clamped)
      DOUBLE PRECISION EPSS               ! Small number to prevent division by zero
      PARAMETER (EPSS = 2.22D-16)         ! Approximate Error parameter

CC     Local Interpolation Variables (For Stability Lookup)
C      INTEGER :: NDXZPL
C      DOUBLE PRECISION :: TGPM, PTPM      

C     ------------------------------------------------------------------
C     Cutoff Guard (Stop calculating if Buoyancy is exhausted)
C     ------------------------------------------------------------------
C     Unpack Buoyancy first to check status
C      FBB    = Y_STATE(4)     ! Buoyancy Flux (m^3/s^2)
      FBB = MAX(Y_STATE(4), 0.0D0) ! Prevent negative buoyancy    ! Buoyancy Flux (m^3/s^2)

C     If Buoyancy is zero or negative, the plume has reached equilibrium.
C     We force derivatives to zero to stop the state from changing (Freeze).
      IF (FBB .LE. 0.0D0) THEN
          DYDX_OUT(1) = 0.0D0
          DYDX_OUT(2) = 0.0D0
          DYDX_OUT(3) = 0.0D0
          DYDX_OUT(4) = 0.0D0
          RETURN
      END IF

C     ------------------------------------------------------------------
C     Unpack remaining state variables for readability
C     ------------------------------------------------------------------
C      RMOM00 = Y_STATE(1)     ! Plume Radius (m)
      RMOM00 = MAX(Y_STATE(1), 1.0D-2) ! Min radius 1 cm     ! Plume Radius (m)
C      ZP00   = Y_STATE(2)     ! Plume Centerline Height (m)
      ZP00   = MAX(Y_STATE(2), HS)     ! Force height to be at least stack height   ! Plume Centerline Height (m)
C      QP00   = Y_STATE(3)     ! Vertical Momentum Parameter (approx w * R^2)
      QP00   = MAX(Y_STATE(3), 0.0D0)  ! Force momentum to be upward                ! Vertical Momentum Parameter (approx w * R^2)

C     ------------------------------------------------------------------
C     Get ambient conditions at current plume height Z
C     ------------------------------------------------------------------
CC     Calculate wind speed (UEFFANN) at the current plume height Z
C      CALL REFWS(ZP00, UEFFANN)
C     Calculate wind speed (UEFFANN) and Stability (N22) continuously
      CALL SIMILARITY_WIND((ZP00 + RMOM00)/2.0D0, UEFFANN, N22)

CC     Calculate Local Stability (N^2)
CC     Find vertical grid index for current height ZP00
C      CALL LOCATE(GRIDHT, 1, MXGLVL, ZP00, NDXZPL)
C
CC     If ZP00 is below the first level of GRIDHT, NDXZPL will be 0.
CC     We force it to 1 to extrapolate using the first layer's gradient.
CC     ----------------------------------------------------------
C      IF (NDXZPL .LT. 1) THEN
C          NDXZPL = 1
C      END IF
C
CC     Interpolate Potential Temperature Gradient (dTheta/dz)
C      CALL GINTRP( GRIDHT(NDXZPL), GRIDTG(NDXZPL),
C     &             GRIDHT(NDXZPL+1), GRIDTG(NDXZPL+1), 
C     &             ZP00, TGPM )
C
CC     Interpolate Potential Temperature (Theta)
C      CALL GINTRP( GRIDHT(NDXZPL), GRIDPT(NDXZPL),
C     &             GRIDHT(NDXZPL+1), GRIDPT(NDXZPL+1), 
C     &             ZP00, PTPM )
C
CC     Calculate Brunt-Vaisala Frequency Squared (N^2)
CC     Formula: N^2 = (g / Theta) * (dTheta/dz)
C      IF (PTPM .GT. 0.0D0) THEN
C            N22 = (G / PTPM) * TGPM
C      ELSE
CC           If Potential Temp is invalid (<=0), we cannot determine stability.
CC          Safest fallback is Neutral (0.0).
CC           Using BVF*BVF here forces stability, which might be wrong in CBL.
C            N22 = 0.0D0 
C      END IF      
C     In unstable air (N22 < 0), FBB should not increase. 
C     We clamp the N22 used for decay (N22_DECAY) to 0.0.
C     This ensures dFBB/dx is 0 in unstable air (Conserved), 
C     rather than positive (Growing).
      N22_DECAY = MAX(N22, 0.0D0)

C     ------------------------------------------------------------------
C     Calculate plume velocities
C     ------------------------------------------------------------------
C     UTT: Total Plume Velocity along the trajectory.
C          The constraint equation (UT_FROM_CONSTRAINT) solves for the velocity
C          required to conserve momentum flux. 
C          We take MAX(..., UEFFANN) because the plume cannot move slower 
C          than the wind pushing it.
      UTT = MAX(UT_FROM_CONSTRAINT(RMOM00, UEFFANN, CFT), UEFFANN)
C     Guard against UTT being slower than ambient wind (prevents division by zero/small)
      IF (UTT .LT. UEFFANN + EPSS) UTT = UEFFANN + EPSS

C     WPPP: Vertical Plume Velocity (w).
C           Derived from Momentum parameter QP00: WPPP = QP00 / Area_factor (RMOM00^2).
C           MAX(..., EPSS) prevents crash if Radius is near zero at start.
C           We limit WPPP >= 0. This implies we stop tracking if the plume
C           starts to sink (overshoot behavior is ignored).
      WPPP = MAX(QP00 / MAX(RMOM00*RMOM00, EPSS), 0.D0)
      IF (WPPP .LT. 0.0D0) WPPP = 0.0D0               ! Limit it to being negative

C     ------------------------------------------------------------------
C     Governing Equations (dY/dx)
C     ------------------------------------------------------------------
C     Conservation of Mass (Radius Growth)
C     dR/dx: How fast the plume expands.
C     ALPHAM term: Shear entrainment (Jet behavior)
C     BETA1 term : Buoyancy entrainment (Plume behavior)
C      DYDX_OUT(1) = ALPHAM*(1.D0 - UEFFANN/UTT) + BETA1*(WPPP/UTT)
C     For turbine/turbofan engines with high exit velocity, the jet entrainment
      IF (BYPR .GT. 0.0D0) THEN
            DYDX_OUT(1) = ALPHAM*((UTT - UEFFANN)*
     &              (2.0D0*UTT - UEFFANN) / 
     &              MAX(UTT*UTT, 1.0D-20) ) + 
     &              BETA1 * (WPPP / UTT)
      ELSE
            DYDX_OUT(1) = BETA1 * (WPPP / UTT)
      END IF
C     Geometry (Trajectory)
C     dZ/dx: Change in height per unit distance.
C     Slope = Rise / Run = Vertical Velocity / Total Velocity
      DYDX_OUT(2) = WPPP / UTT

C     Conservation of Vertical Momentum
C     dQ/dx: Change in vertical momentum.
C     According to Newton's law: Change in momentum = Buoyancy Force / Velocity
C     (FBB is the buoyancy flux driving the rise)
      DYDX_OUT(3) = FBB / UTT

C     Conservation of Buoyancy (Energy)
C     dFBB/dx: Loss of buoyancy due to stratification.
C     In a stable atmosphere (N22 > 0), the plume entrains heavier air, reducing FBB.
C     This formulation couples the loss to the radius growth rate (DYDX_OUT(1)).
C     Standard approx: dFBB/dx ~ -N22 * QP00 / UT.
      DYDX_OUT(4) = -(N22_DECAY/BETA1)*RMOM00*RMOM00*DYDX_OUT(1)

      RETURN
      END

      DOUBLE PRECISION FUNCTION UT_FROM_CONSTRAINT(RAC, UAC, CAC)
C***********************************************************************
C             UT_FROM_CONSTRAINT function of the AMS/EPA Regulatory Model - AERMOD
C
C   PURPOSE: Calculates the Total Plume Velocity (UT) by solving a quadratic 
C       constraint equation: UT * (UT - UAC) = CAC / RAC^2.
C
C   PROGRAMMER: Gavendra Pandey, UNC-IE, Chapel Hill, NC, USA &
C               Akula Venkatram, UC-Riverside, CA, USA
C
C   DATE:    January 29, 2026
C
C   INPUTS:  
C       RAC: Current Plume Radius (m)
C       UAC: Ambient Wind Speed (m/s)
C       CAC: Constraint Constant (Velocity closure parameter, m^4/s^2)
C
C   OUTPUTS: UT_FROM_CONSTRAINT : The calculated total velocity (m/s).
C
C   CALLED FROM:  RHS_IN_X
C
C***********************************************************************
      IMPLICIT NONE

      DOUBLE PRECISION RAC, UAC, CAC
      DOUBLE PRECISION RRC, DISCC, ROOTC
      DOUBLE PRECISION EPSC
      PARAMETER (EPSC = 2.22D-16)                ! Change it from 1.0D-12 to 2.22D-16

C     Safety Check for Radius
C     Avoid division by zero if RAC is 0 (at the very start of simulation).
C     Using DSQRT(EPSC) ensures RRC^2 is at least EPSC.
      RRC = MAX(RAC, DSQRT(EPSC))

C     Calculate Discriminant
C     We are solving the quadratic equation: UT^2 - UAC*UT - (CAC/RRC^2) = 0
C     Using the quadratic formula: x = (-b + sqrt(b^2 - 4ac)) / 2a
C     Here: a=1, b=-UAC, c=-(CAC/RRC^2)
C     Discriminant = b^2 - 4ac = UAC^2 + 4 * (CAC/RRC^2)
      DISCC = MAX(UAC*UAC + 4.D0*CAC/(RRC*RRC), 0.D0)
      ROOTC = DSQRT(DISCC)

C     Calculate Primary Root
C     We generally want the positive root (velocity must be positive).
      UT_FROM_CONSTRAINT = 0.5D0*(UAC + ROOTC)
C     Defensive Programming (Edge Cases)
C     If the primary root is weirdly non-positive (unlikely with positive CAC),
C     try the other root of the quadratic.
      IF (UT_FROM_CONSTRAINT .LE. 0.D0) THEN
            UT_FROM_CONSTRAINT = 0.5D0*(UAC - ROOTC)
      ENDIF

C     If it is STILL non-positive (e.g., if physics broke down),
C     force it to be at least the ambient wind speed + a tiny bit.
      IF (UT_FROM_CONSTRAINT .LE. 0.D0) THEN
            UT_FROM_CONSTRAINT = MAX(UAC, 0.D0) + DSQRT(EPSC)
      ENDIF

      RETURN
      END

      SUBROUTINE SIMILARITY_WIND(ZPPP, U_OUT, N2_OUT)
C***********************************************************************
C            SIMILARITY_WIND module of the AMS/EPA Regulatory Model - AERMOD
C
C   PURPOSE: Calculates Wind and Stability using AERMOD's vertical profiles.
C
C   PROGRAMMER: Gavendra Pandey, UNC-IE, Chapel Hill, NC, USA
C
C   DATE:    April 24, 2026
C
C   INPUTS:  
C       ZPPP: Current Plume Height (m)
C
C   OUTPUTS: U_OUT : Wind speed U_OUT at ZPPP; (m/s),
C                    Squared Brunt Vaisala Frequency N2_OUT at ZPPP; (1/s2)
C
C   CALLED FROM:  PLUME_TRAJ, RHS_IN_X
C
C***********************************************************************
      USE MAIN1
      IMPLICIT NONE

      DOUBLE PRECISION, INTENT(IN)  :: ZPPP   ! Current Plume Height
      DOUBLE PRECISION, INTENT(OUT) :: U_OUT  ! Wind Speed
      DOUBLE PRECISION, INTENT(OUT) :: N2_OUT ! Stability (N^2)

C     Local variables
      INTEGER :: NDX
      DOUBLE PRECISION :: TG_LOC, PT_LOC

C     Find the index in the AERMOD meteorological grid for height ZPPP
      CALL LOCATE(GRIDHT, 1, MXGLVL, ZPPP, NDX)
      
C     Ensure index is within bounds (1 to MXGLVL-1)
      NDX = MAX(1, MIN(NDX, MXGLVL - 1))

C     Interpolate Potential Temperature Gradient (dTheta/dz)
      CALL GINTRP(GRIDHT(NDX), GRIDTG(NDX), GRIDHT(NDX+1), 
     &            GRIDTG(NDX+1), ZPPP, TG_LOC)

C     Interpolate Potential Temperature (Theta)
      CALL GINTRP(GRIDHT(NDX), GRIDPT(NDX), GRIDHT(NDX+1), 
     &            GRIDPT(NDX+1), ZPPP, PT_LOC)

C     Calculate Wind Speed at ZPPP (Standard AERMOD profile)
      CALL REFWS(ZPPP, U_OUT)

C     Calculate N^2 (Stability)
      IF (STABLE .AND. PT_LOC .GT. 0.0D0) THEN
          N2_OUT = (G / PT_LOC) * TG_LOC
      ELSE
          N2_OUT = 0.0D0
      END IF

      RETURN
      END