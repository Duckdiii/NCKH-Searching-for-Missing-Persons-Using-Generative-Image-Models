import React from 'react';
import { Check, Lock } from 'lucide-react';

export type WizardStep = 'restore' | 'generate' | 'results';

export interface WizardStepperProps {
  currentStep: WizardStep;
  completedSteps: WizardStep[];
  onStepClick: (step: WizardStep) => void;
}

interface StepMeta {
  id: WizardStep;
  num: number;
  label: string;
  sublabel: string;
}

const STEPS: StepMeta[] = [
  { id: 'restore', num: 1, label: 'Khôi phục ảnh cũ', sublabel: 'CodeFormer & Chọn mặt' },
  { id: 'generate', num: 2, label: 'Sinh ảnh & Đối soát', sublabel: 'FADING Inversion & Search' },
  { id: 'results', num: 3, label: 'Kết quả đối soát', sublabel: 'Bằng chứng & Đánh giá' },
];

export const WizardStepper: React.FC<WizardStepperProps> = ({
  currentStep,
  completedSteps,
  onStepClick,
}) => {
  return (
    <div className="bg-white border border-[#E5E7EB] rounded-xl p-4 shadow-xs">
      <div className="flex items-center justify-between relative">
        {STEPS.map((step, idx) => {
          const isCompleted = completedSteps.includes(step.id);
          const isCurrent = currentStep === step.id;
          const isLocked = !isCompleted && !isCurrent;
          const canClick = isCompleted && !isCurrent;

          return (
            <React.Fragment key={step.id}>
              {idx > 0 && (
                <div
                  data-testid={`connector-${idx}`}
                  className={`flex-1 h-0.5 mx-3 transition-colors duration-300 ${
                    completedSteps.includes(STEPS[idx - 1].id)
                      ? 'bg-[#3B82C7]'
                      : 'bg-[#E5E7EB]'
                  }`}
                />
              )}

              <button
                type="button"
                data-testid={`wizard-step-${step.id}`}
                aria-label={`Bước ${step.num}: ${step.label}`}
                disabled={isLocked || isCurrent}
                onClick={() => {
                  if (canClick) {
                    onStepClick(step.id);
                  }
                }}
                className={`flex items-center gap-3 text-left transition-all px-2.5 py-1.5 rounded-xl ${
                  canClick
                    ? 'cursor-pointer hover:bg-[#F9FAFB] hover:ring-1 hover:ring-[#3B82C7]/40 group'
                    : isCurrent
                    ? 'cursor-default'
                    : 'cursor-not-allowed opacity-40 pointer-events-none'
                }`}
              >
                <div
                  className={`w-9 h-9 rounded-full flex items-center justify-center font-bold text-xs transition-all border ${
                    isCurrent
                      ? 'bg-[#E8804A] border-[#E8804A] text-white shadow-sm ring-2 ring-[#E8804A]/30'
                      : isCompleted
                      ? 'bg-[#EFF6FF] border-[#3B82C7] text-[#3B82C7] group-hover:scale-105 group-hover:bg-[#DBEAFE]'
                      : 'bg-[#F9FAFB] border-[#E5E7EB] text-[#9CA3AF]'
                  }`}
                >
                  {isCompleted ? (
                    <Check className="w-4 h-4 text-[#3B82C7] stroke-[2.5]" />
                  ) : isLocked ? (
                    <Lock className="w-3.5 h-3.5 text-[#9CA3AF]" />
                  ) : (
                    <span>{step.num}</span>
                  )}
                </div>

                <div className="hidden sm:block">
                  <p
                    className={`text-xs font-semibold ${
                      isCurrent
                        ? 'text-[#111827]'
                        : isCompleted
                        ? 'text-[#3B82C7]'
                        : 'text-[#9CA3AF]'
                    }`}
                  >
                    {step.label}
                  </p>
                  <p className="text-[10px] text-[#6B7280] mt-0.5">{step.sublabel}</p>
                </div>
              </button>
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};
