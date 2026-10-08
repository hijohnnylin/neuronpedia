-- AlterTable
ALTER TABLE "JlensShare" ADD COLUMN     "lensColumns" TEXT[] DEFAULT ARRAY[]::TEXT[];

-- Backfill: the old lens tab as a column set. DIFF showed both token lenses.
UPDATE "JlensShare"
SET "lensColumns" = CASE "activeLensModeTab"
  WHEN 'DIFF' THEN ARRAY['JACOBIAN_LENS', 'LOGIT_LENS']::TEXT[]
  WHEN 'LOGIT_LENS' THEN ARRAY['LOGIT_LENS']::TEXT[]
  ELSE ARRAY['JACOBIAN_LENS']::TEXT[]
END;
