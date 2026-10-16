# v1 origin: tasks/permit-forge/instances/003bd328a392-s47/tests/instrument.py
"""Add read-only statistical observation to the interactive and verifier AERMOD builds.

No model inputs, arithmetic or control flow are changed. Retains only annual
means, annual regulatory daily ranks, and annual second-highest 3-hour values.
"""
from pathlib import Path
import sys
p = Path(sys.argv[1]) / 'aermod.f'
s = p.read_text()
for old, new in [
    ('                  CALL HIVALS', '                  CALL PF_SHORT\n                  CALL HIVALS'),
    ('            CALL PERAVE\nC ---       Check for ANNUAL POSTFILE',
     '            CALL PERAVE\n            CALL PF_YEAR\nC ---       Check for ANNUAL POSTFILE')]:
    assert s.count(old) == 1, old
    s = s.replace(old, new)
s += '''
      SUBROUTINE PF_SHORT
      USE MAIN1
      IMPLICIT NONE
      DOUBLE PRECISION, ALLOCATABLE, SAVE :: FIRST(:), SECOND(:)
      DOUBLE PRECISION, ALLOCATABLE, SAVE :: DAYMAX(:), TOP4(:,:)
      DOUBLE PRECISION PFVALUE,PFSWAP
      INTEGER PFREC,PFK
      IF (.NOT.ALLOCATED(FIRST)) THEN
         ALLOCATE(FIRST(NUMREC),SECOND(NUMREC))
         ALLOCATE(DAYMAX(NUMREC),TOP4(NUMREC,4))
         DAYMAX=0.0D0
         TOP4=0.0D0
         FIRST=0.0D0
         SECOND=0.0D0
      ENDIF
      IF (POLLUT.EQ.'SO2'.AND.KAVE(IAVE).EQ.1) THEN
         DO PFREC=1,NUMREC
            DAYMAX(PFREC)=MAX(DAYMAX(PFREC),AVEVAL(PFREC,1,IAVE,1))
         ENDDO
         IF (IHOUR.EQ.24) THEN
            DO PFREC=1,NUMREC
               PFVALUE=DAYMAX(PFREC)
               DO PFK=1,4
                  IF (PFVALUE.GT.TOP4(PFREC,PFK)) THEN
                     PFSWAP=TOP4(PFREC,PFK)
                     TOP4(PFREC,PFK)=PFVALUE
                     PFVALUE=PFSWAP
                  ENDIF
               ENDDO
            ENDDO
            DAYMAX=0.0D0
         ENDIF
      ENDIF
      IF (KAVE(IAVE).NE.3) RETURN
      DO PFREC=1,NUMREC
         IF (AVEVAL(PFREC,1,IAVE,1).GE.FIRST(PFREC)) THEN
            SECOND(PFREC)=FIRST(PFREC)
            FIRST(PFREC)=AVEVAL(PFREC,1,IAVE,1)
         ELSE IF (AVEVAL(PFREC,1,IAVE,1).GT.SECOND(PFREC)) THEN
            SECOND(PFREC)=AVEVAL(PFREC,1,IAVE,1)
         ENDIF
      ENDDO
      RETURN
      ENTRY PF_THREE_YEAR
      IF (.NOT.ALLOCATED(FIRST)) RETURN
      OPEN(UNIT=991,FILE='pf_stats.txt',POSITION='APPEND')
      IF (POLLUT.EQ.'SO2'.AND..NOT.SO2AVE) THEN
         DO PFREC=1,NUMREC
            WRITE(991,'(A,2I8,ES26.17)') 'DAILY ',NUMYRS,PFREC,
     &                                 TOP4(PFREC,4)
         ENDDO
      ENDIF
      TOP4=0.0D0
      IF (ANY(KAVE(1:NUMAVE).EQ.3)) THEN
      DO PFREC=1,NUMREC
         WRITE(991,'(A,2I8,ES26.17)') 'THREE ',NUMYRS,PFREC,
     &                              SECOND(PFREC)
      ENDDO
      ENDIF
      CLOSE(991)
      FIRST=0.0D0
      SECOND=0.0D0
      END

      SUBROUTINE PF_YEAR
      USE MAIN1
      IMPLICIT NONE
      INTEGER PFREC,PFRANK
      OPEN(UNIT=991,FILE='pf_stats.txt',POSITION='APPEND')
      DO PFREC=1,NUMREC
         WRITE(991,'(A,2I8,ES26.17)') 'ANNUAL ',NUMYRS,PFREC,
     &                              ANNVAL(PFREC,1,1)
      ENDDO
      IF (PM25AVE.OR.NO2AVE.OR.SO2AVE) THEN
         PFRANK=8
         IF (SO2AVE) PFRANK=4
         DO PFREC=1,NUMREC
            WRITE(991,'(A,2I8,ES26.17)') 'DAILY ',NUMYRS,PFREC,
     &                                 HIMXDLY(PFREC,1,PFRANK)
         ENDDO
      ENDIF
      CLOSE(991)
      CALL PF_THREE_YEAR
      END
'''
p.write_text(s)
