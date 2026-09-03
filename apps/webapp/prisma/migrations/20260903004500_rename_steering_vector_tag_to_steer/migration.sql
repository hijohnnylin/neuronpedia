-- `steering-vector` said "vector" twice: the tag only ever sits on a `Vector`, so the name is
-- `steer` -- what the direction is for, in the same shape as `axis` and `probe`.
--
-- An UPDATE rather than an insert-and-delete: `VectorTagOnVector_tagName_fkey` is ON UPDATE CASCADE,
-- so any vector wearing the tag follows the rename. None does today, and this stays right when one
-- does.
UPDATE "VectorTag"
SET "name" = 'steer', "displayName" = 'Steering'
WHERE "name" = 'steering-vector';
