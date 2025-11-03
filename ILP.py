from hessian import hutchinson
from ilp_utils import *
import numpy as np
from pulp import *
import pulp

def ilp_main_bops(model, modified_hawq = True, bops_limit_factor = 0.5):
  model.eval()
  model = model.to('cpu')
  l2_norms = {}
  #If 2-8 bits or just [4,8] bits
  if modified_hawq:
    quantisation_modes = range(2,9)
  else:
    quantisation_modes = [4,8]
    
  for bitwidth in quantisation_modes:
    #L2 norms of each layer
    l2_differences = l2_diff_conv_linear(model, bit_a=bitwidth, device=device)
    l2_norms[bitwidth] = {}
    for layer_name, l2_val in l2_differences.items():
        l2_norms[bitwidth][layer_name] = l2_val
   

  #Bit ops for every precision
  bops = {}
  bitops = get_bitops(model, w_bits=1, a_bits=8)
  bops = np.array(list(bitops.values()))
  bops_bit = {}
  for i in quantisation_modes:
    bops_bit[i] = bops * i  / 1000000     #Scaled to prevent blowing up of values
    
  #Finding Hessian and number of parameters of every layer
  dummyinput = torch.randn(1, 3, 32, 32) #random image input
  dummytarget = torch.tensor([3]) # random target class
  print("Calculation Hutchinsons Trace......")
  results_hessian = hutchinson(model, dummyinput, dummytarget, num_samples=10)
  print("Finished Calculating Trace")
  Hutchinson_trace = []
  parameters = []
  for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
      Hutchinson_trace.append(results_hessian[f"{name}.weight"]['normalized_trace'])
      parameters.append(results_hessian[f"{name}.weight"]['num_params'])
  Hutchinson_trace = np.array(Hutchinson_trace)
  parameters = np.array(parameters) / 1024 / 1024
  num_layers = Hutchinson_trace.shape[0]

  limit_selecting_layer = 4
  
  bops_limit = np.sum(bops_bit[limit_selecting_layer]) + (np.sum(bops_bit[8]) - np.sum(bops_bit[limit_selecting_layer])) * bops_limit_factor
  
  if modified_hawq: 
    delta_weights_diff = {}
    for i in quantisation_modes:
      delta_weights_diff[i] = np.array(list(l2_norms[i].values())) - np.array(list(l2_norms[8].values()))
    sensitivity_difference = {}
    for i in quantisation_modes:
      sensitivity_difference[i] = Hutchinson_trace * delta_weights_diff[i]
    

    x = {i: {j: LpVariable(f"x{i}_{j}", cat='Binary') for j in quantisation_modes} for i in range(num_layers)}

    prob = LpProblem("BOPS", LpMinimize)

    # A layer has to at least have one 1
    for i in range(num_layers):
      prob += lpSum([ x[i][j] for j in quantisation_modes ]) == 1

    # add bops constraint
    prob += lpSum(bops_bit[j][i] * x[i][j] for i in range(num_layers) for j in quantisation_modes) <= bops_limit

    # Sensitivity Constraint
    prob += sum( [sensitivity_difference[j][i] * x[i][j] for i in range(num_layers) for j in quantisation_modes] )

    # solve the problem
    status = prob.solve(GLPK_CMD(msg=1, options=["--tmlim", "10000","--simplex"]))

    # get the result
    LpStatus[status]

    result = []
    for i in range(num_layers):
      for j in quantisation_modes:
        if value(x[i][j]) == 1:
          result.append(j)
    result = np.array(result)


#the original hawq implementation with only 4 and 8 bits
  else:
    delta_weights_4bit_square = np.array(list(l2_norms[4].values()))
    delta_weights_8bit_square = np.array(list(l2_norms[8].values()))
    variable = {}
    for i in range(num_layers):
        variable[f"x{i}"] = LpVariable(f"x{i}", 1, 2, cat=LpInteger)
    prob = LpProblem("BOPS", LpMinimize)

    bops_difference_between_4_8 = bops_bit[8] - bops_bit[4]

    # add bops constraint
    prob += sum([ bops_bit[4][i] + (variable[f"x{i}"] - 1) * bops_difference_between_4_8[i] for i in range(num_layers) ]) <= bops_limit # if 4 bit, x - 1 = 0, we use bops_4, if 8 bit, x-1=2, we use bops_diff + bos_4 = bops_8


    sensitivity_difference_between_4_8 = Hutchinson_trace * ( delta_weights_8bit_square  - delta_weights_4bit_square ) # here is the sensitivity different between 4 and 8

    prob += sum( [ (variable[f"x{i}"] - 1) * sensitivity_difference_between_4_8[i] for i in range(num_layers) ] )

    # solve the problem
    status = prob.solve(GLPK_CMD(msg=1, options=["--tmlim", "10000","--simplex"]))

    # get the result
    LpStatus[status]

    result = []
    for i in range(num_layers):
        result.append(value(variable[f"x{i}"]))
    result = np.array(result)*4

  bitmap_dict = {}
  i = 0
  for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
      bitmap_dict[name] = int(result[i])
      i += 1

  total_bops = 0
  for i in range(num_layers):
    total_bops += bops_bit[result[i]][i]
  
  total_bops = round(total_bops*1000000)
  return bitmap_dict, total_bops

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HAWQ-V3 ILP Quantization")
    parser.add_argument("--modified_hawq", type=lambda x: x.lower() == 'true', default=True, help="Use modified HAWQ (True/False)")
    parser.add_argument("--bops_limit_factor", type=float, default=0.5, help="BOPs limit factor")
    args = parser.parse_args()

    # Example: using pretrained ResNet18
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bitmap_dict, total_bops = ilp_main_bops(
        model=model,
        device=device,
        modified_hawq=args.modified_hawq,
        bops_limit_factor=args.bops_limit_factor
    )

    print("\n=== ILP Quantization Result ===")
    for layer, bits in bitmap_dict.items():
        print(f"{layer:40s} → {bits}-bit")
    print(f"\nTotal BOPs (approx): {total_bops:,}")
