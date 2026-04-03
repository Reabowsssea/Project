#include "Holodeck.h"
#include "HoveringAUVControlThrusters1.h"


UHoveringAUVControlThrusters1::UHoveringAUVControlThrusters1(const FObjectInitializer& ObjectInitializer) :
		Super(ObjectInitializer) {}

void UHoveringAUVControlThrusters1::Execute(void* const CommandArray, void* const InputCommand, float DeltaSeconds) {
	if (HoveringAUV == nullptr) {
		HoveringAUV = static_cast<AHoveringAUV*>(HoveringAUVController->GetPawn());
		if (HoveringAUV == nullptr) {
			UE_LOG(LogHolodeck, Error, TEXT("UHoveringAUVControlThrusters1 couldn't get HoveringAUV reference"));
			return;
		}
		
		HoveringAUV->EnableDamping();
	}

	float* InputCommandFloat = static_cast<float*>(InputCommand);
	float* CommandArrayFloat = static_cast<float*>(CommandArray);

	HoveringAUV->ApplyBuoyantForce();
	HoveringAUV->ApplyThrusters(InputCommandFloat);

	// Zero out the physics based controller
	for(int i=0; i<6; i++){
		CommandArrayFloat[i] = 0;
	}
}