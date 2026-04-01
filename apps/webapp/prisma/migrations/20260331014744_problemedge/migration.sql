-- AlterTable
ALTER TABLE "ProblemEdge" ADD COLUMN     "approvalState" "ProblemNodeApprovalState" NOT NULL DEFAULT 'PENDING',
ADD COLUMN     "approverId" TEXT;

-- AddForeignKey
ALTER TABLE "ProblemEdge" ADD CONSTRAINT "ProblemEdge_approverId_fkey" FOREIGN KEY ("approverId") REFERENCES "User"("id") ON DELETE SET NULL ON UPDATE CASCADE;
