local torch = require("torch")
local nn = require("nn")
local image = require("image")
local paths = require("paths")

local assert = assert
local io = io
local loadfile = loadfile
local math = math
local pairs = pairs
local print = print
local setfenv = setfenv
local unpack = unpack

module("dream")

-- Helper print function
function printf(fmt, ...)
	print(fmt:format(...))
end

config = {
	-- Size of images for the network
	INPUT_WIDTH = 224;
	INPUT_HEIGHT = 224;

	-- I used this: https://github.com/soumith/inception.torch
	-- There is a deeper version here: https://github.com/Moodstocks/inception-v3.torch
	model_path = './inception.t7';

	-- the reference model has channels in BGR order instead of RGB: Set this to true to swap the red/blue channels.
	-- Setting this to false by default because its output with the sample image (output-rgb-chunked.jpg) looks most similar to the original deepdream output
	use_bgr = true;

	imagenet_mean = { -- ImageNet mean, training set dependent (BGR order?)
		104.0,
		116.0,
		122.0
	};

	-- math.max(unpack(imagenet_mean))
	imagenet_min = -122;
	-- 255 + imagenet_min
	imagenet_max = 255 - 122;
	
	-- If use_whole_image is true, the entire image, whatever its size, is
	-- passed into net:forward(), as in the EladHoffer implementation.
	-- This seems to work well at upper layers...but that will lead to the
	-- output layers being larger than 1x1000
	-- If use_whole_image is false, the image is fed into the net in 
	-- IMAGE_WIDTH x IMAGE_HEIGHT pieces
	use_whole_image = false;

	-- File to use if arg[1] is not provided
	input_file = "input.jpg";

	-- Final file is saved to output.jpg
	output_file = "output.jpg";

	-- Maximum value of the larger dimension (nil to not scale)
	max_image_size = 440;

	-- function to call after each step. Receives octave, i, img.
	vis_callback = function(options)
		local octave = options.octave
		local i = options.i
		local img = options.img
		-- reuse the window. Has the unfortunate side-effect of keeping the size of the original image...
		vis_window = showarray(img, vis_window)
		-- TODO FIXME: calling image.save is causing img to be corrupted...even if we save a clone
		-- if i % 10 > 0 then return end
		-- local file = ("frame%02d.%04d.jpg"):format(octave, i)
		-- printf("Writing image to %s...", file)
		-- image.save(file, img)
		-- image.save(file, img:clone())
	end
}

if paths.filep("config.lua") then
	printf("Loading config from config.lua...")
	local configFunc = assert(loadfile("config.lua"))
	setfenv(configFunc, config)
	configFunc()
end


-- TODO: handle sub-layers
layer_names = {
	"conv2d0",
	"Relu1",
	"MaxPool2",
	"LRN3",
	"conv2d1",
	"Relu5",
	"conv2d2",
	"Relu7",
	"LRN8",
	"MaxPool9",
	"inception_3a",
	"inception_3b",
	"MaxPool38",
	"inception_4a",
	"inception_4b",
	"inception_4c",
	"inception_4d",
	"inception_4e",
	"MaxPool109",
	"inception_5a",
	"inception_5b",
	"AveragePool138",
	"view_1024_3",
	"FC153",
	"narrow_2_2_1000",
	"softmax2" -- softmax 
}

-- map from top-level layer name to layer number
layer_numbers = { }

for k, v in pairs(layer_names) do
	layer_numbers[v] = k
end

--[[*
	 show an image (undoing any pre-processing)
	 @tparam Tensor[3][W][H] img the network-formatted image to display
	 @tparam userdata win optional window to show the image in
	 @treturn userdata the window where the image was displayed
--]]
function showarray(img, win)
	swap_red_blue(img)
	local ret = image.display({
		image = img,
		win = win,
		min = imagenet_min,
		max = imagenet_max
	})
	swap_red_blue(img)
	return ret
end

-- ??? preprocess/deprocess ???
--[[*
	Take an external-format image (RGB, values ranging from 0..1) and
	return the equivalent network-format image (normalized, with red/
	blue channels swapped if applicable).
	@tparam Tensor[3][W][H] img the original image (not modified)
	@treturn Tensor[3][W][H] the pre-processed image
--]]
function preprocess(img)
	local ret = img:clone():mul(255):double()
	swap_red_blue(ret)
	local red_index = config.use_bgr and 3 or 1
	local blue_index = config.use_bgr and 1 or 3
	local green_index = 2
	ret[red_index] = ret[red_index] - config.imagenet_mean[3]
	ret[green_index] = ret[green_index] - config.imagenet_mean[2]
	ret[blue_index] = ret[blue_index] - config.imagenet_mean[1]
	return ret
end

--[[*
	If use_bgr is set, swap the red and blue channels of the image.
	Modifies the image itself.
	@tparam Tensor[3][W][H] img the image to modify
	@treturn Tensor[3][W][H] img itself, after swapping channels
--]]
function swap_red_blue(img)
	if not config.use_bgr then return img end
	local reds = torch.Tensor(img:size(2), img:size(3))
	reds[{}] = img[1]
	img[1] = img[3]
	img[3] = reds[{}]
	return img
end

--[[*
	Take a network-format image (normalized, with red/blue channels 
	swapped if applicable) abd return the equivalent external-format
	image (RGB, values ranging from 0..1).
	@tparam Tensor[3][W][H] img the pre-processed image (not modified)
	@treturn Tensor[3][W][H] the de-processed image
--]]
function deprocess(img)
	local ret = img:clone()
	local red_index = config.use_bgr and 3 or 1
	local blue_index = config.use_bgr and 1 or 3
	local green_index = 2
	ret[red_index] = ret[red_index] + config.imagenet_mean[3]
	ret[green_index] = ret[green_index] + config.imagenet_mean[2]
	ret[blue_index] = ret[blue_index] + config.imagenet_mean[1]
	return swap_red_blue(ret:mul(1/255.0):clamp(0, 1))
end

-- About objective functions:
--	net:backward(...) takes the value of dLoss/dOutput[x]
--  So, for an L2 norm with expected values E[x], the loss function would be
--		L2(E, out) = sum[x] 1/2 (E[x] - out[x])^2
--  Then dL2(E, out)/d(out[x]) = out[x] - E[x]
-- Note that DeepDream does a gradient *ascent*, rather than the usual gradient *descent*
-- (i.e. in[x] = in[x] + learningRate * dL/d(in[x])), so that the loss function is maximized instead of minimized.

--[[*
	the default deepdream objective function
	@tparam Tensor dst the output of net:forward()
	@treturn Tensor dst itself (corresponds to an L2 norm with an all-zero expected value)
--]]
function objective_L2(dst)
	return dst
end

--[[*
	Helper function to get the layer number given a layer name
	TODO: Support sub-layers
	@tparam string layer the layer name
	@treturn number the index of the layer
--]]
function get_layer_number(layer)
	return layer_numbers[layer]
end

-- Cache of truncated networks: map of layer name -> network truncated at that layer
local layerCache = {}

--[[*
	Truncate a network at the given layer.
	@tparam nn.Sequential net the network to truncate
	@tparam string layer the name of the layer TODO support sub-layers
	@treturn nn.Sequential the truncated network ending at the requested layer
--]]
function truncate_network(net, layer)
	local netCache = layerCache[net] or {}
	if not layerCache[net] then layerCache[net] = netCache end
	local cached = netCache[layer]
	if cached then return cached end
	-- TODO:handle sub-layers
	local ret = nn.Sequential()
	local layer_number = get_layer_number(layer)
	for i = 1, layer_number do
		ret:add(net:get(i))
	end
	netCache[layer] = ret
	return ret
end

--[[*
	Utility function to clamp the network-format image's values to the
	maximum range. Modifies the image itself.
	@tparam Tensor[3][W][H] img the image to clamp.
	@treturn Tensor[3][W][H] the image, after clamping.
--]]
function clamp(img)
	local red_index = config.use_bgr and 3 or 1
	local blue_index = config.use_bgr and 1 or 3
	local green_index = 2
	img[red_index]:clamp(-config.imagenet_mean[3], 255 - config.imagenet_mean[3])
	img[green_index]:clamp(-config.imagenet_mean[2], 255 - config.imagenet_mean[2])
	img[blue_index]:clamp(-config.imagenet_mean[1], 255 - config.imagenet_mean[1])
	return img
end

--[[*
	Basic gradient ascent step.
	Passes the entire image into the network, regardless of its 
	dimensions.
	@tparam table options named parameters
	@tparam nn.Sequential options.net the network
	@tparam Tensor[3][W][H] options.src the pre-processed image
	@tparam number options.step_size the step size. Default 1.5
	@tparam string options.end_layer the layer to stop at. Default "inception_4c". TODO: Implement sub-layers
	@tparam number options.jitter the jitter amount. Default 32. TODO: IMPLEMENT
	@tparam boolean options.clip if true, clamp the values to the valid range
	@tparam function options.objective the objective function.
	@treturn Tensor[3][W][H] the modified image
--]]
function make_step_whole_image(options)
	local defaults = {
		jitter = 32,
		step_size = 1.5,
		end_layer = "inception_4c", -- note that nodes within the inception nodes are not addressable
		clip = true,
		objective = objective_L2
	}
	for k, v in pairs(defaults) do
		if options[k] == nil then
			options[k] = v
		end
	end
	assert(options.net, "options.net is required")
	assert(options.src, "options.src is required")
	local net = truncate_network(options.net, options.end_layer)
	-- here we replace jitter by choosing a random INPUT_WIDTH x INPUT_HEIGHT sub-image to process.
	local W, H = options.src:size(3), options.src:size(2)
	local ox = math.random(0, 2 * options.jitter) - options.jitter
	local oy = math.random(0, 2 * options.jitter) - options.jitter
	-- Create a view of a single-image batch
	local src = options.src:view(1, unpack(options.src:size():totable()))
	net:evaluate()
	local output = net:forward(src)
	-- Some nets require training to be set to enable calculating the derivatives
	net:training()
	local dL_d_output = options.objective(output)
	local dL_d_src = net:updateGradInput(src, output)
	-- src[rgb][y][x] += step_size * normalize(dL/d(src[y][x]))
	local normalized = dL_d_src:mul(1/torch.abs(dL_d_src):mean())
	normalized:mul(options.step_size)
	src:add(normalized)

	if clip then
		clamp(src)
	end

	return options.src
end

-- simulating np.roll, which does a translation with wrap-around
-- returns a copy
-- TODO: support negative values
function jitter(img, ox, oy)
--[[
		/----\		-\/---		8||567
		|1234|		4||123		_/\___
		|5678| ==>	8||567	==>	-\/---
		\____/		_/\___		4||123
--]]
	local _, H, W = unpack(img:size():totable())
	local topLeft = torch.Tensor(_, H - oy, W - ox)
	local topRight = torch.Tensor(_, H - oy, ox)
	local bottomLeft = torch.Tensor(_, oy, W - ox)
	local bottomRight = torch.Tensor(_, oy, ox)
	local ret = torch.Tensor(_, H, W):zero()
	
	topLeft[{}] = img[{{}, {1, H - oy }, { 1, W - ox }}]
	topRight[{}] = img[{{}, {1, H - oy }, { W - ox + 1, W}}]
	bottomLeft[{}] = img[{{}, {H - oy + 1, H}, {1, W - ox}}]
	bottomRight[{}] = img[{{}, {H - oy + 1, H}, {W - ox + 1, W}}]
	
	-- top left to the bottom right, offset by ox, H - oy
	ret[{{}, {oy + 1, H}, {ox + 1, W}}] = topLeft
	-- top right to bottom left, offset by 0, H - oy
	ret[{{}, {oy + 1, H}, {1, ox}}] = topRight
	-- bottom left to top right, offset by ox, 0
	ret[{{}, {1, oy}, {ox + 1, W}}] = bottomLeft
	-- bottom right to top left, offset by 0, 0
	ret[{{}, {1, oy}, {1, ox}}] = bottomRight

	return ret
end

-- undoes jitter (returns a copy)
function unjitter(img, ox, oy)
--[[
		/----\		-\/---		8||567
		|1234|		4||123		_/\___
		|5678| ==>	8||567	==>	-\/---
		\____/		_/\___		4||123
		
		8||567		-\/---		/---*-\
		_/\___		4||123		|123*4|
		-\/--- ==>	~~~~~~	==> ~~~~~~~
		4||123		8||567		|567*8|
					-/\___		\---*-/
--]]
	local _, H, W = unpack(img:size():totable())
	local topLeft = torch.Tensor(_, H - oy, W - ox)
	local topRight = torch.Tensor(_, H - oy, ox)
	local bottomLeft = torch.Tensor(_, oy, W - ox)
	local bottomRight = torch.Tensor(_, oy, ox)
	local ret = torch.Tensor(_, H, W):zero()
	
	topLeft[{}] = img[{{}, {oy + 1, H}, { ox + 1, W }}]
	topRight[{}] = img[{{}, {oy + 1, H}, {1, ox}}]
	bottomLeft[{}] = img[{{}, {1, oy}, {ox + 1, W}}]
	bottomRight[{}] = img[{{}, {1, oy}, {1, ox}}]
	
	ret[{{}, {1, H - oy}, {1, W - ox}}] = topLeft
	ret[{{}, {1, H - oy}, {W - ox + 1, W}}] = topRight
	ret[{{}, {H - oy + 1, H}, {1, W - ox}}] = bottomLeft
	ret[{{}, {H - oy + 1, H}, {W - ox + 1, W}}] = bottomRight
	return ret
end

--[[*
	Basic gradient ascent step.
	Run a series of steps on random 224x224 chunks of the image
	@tparam table options named parameters
	@tparam nn.Sequential options.net the network
	@tparam Tensor[3][W][H] options.src the pre-processed image
	@tparam number options.step_size the step size. Default 1.5
	@tparam string options.end_layer the layer to stop at. Default "inception_4c". TODO: Implement sub-layers
	@tparam number options.jitter the jitter amount. Default 32. TODO: IMPLEMENT
	@tparam boolean options.clip if true, clamp the values to the valid range
	@tparam function options.objective the objective function.
	@treturn Tensor[3][W][H] the modified image
--]]
function make_step_chunked(options)
	local defaults = {
		step_size = 1.5,
		end_layer = "inception_4c", -- note that nodes within the inception nodes are not addressable
		clip = true,
		objective = objective_L2,
		jitter = 32
	}
	for k, v in pairs(defaults) do
		if options[k] == nil then
			options[k] = v
		end
	end
	assert(options.net, "options.net is required")
	assert(options.src, "options.src is required")
	local jx, jy = 0, 0
	if options.jitter > 0 then
		jx = math.random(1, options.jitter)
		jy = math.random(1, options.jitter)
		options.src = jitter(options.src, jx, jy)
	end

	local net = truncate_network(options.net, options.end_layer)
	-- here we replace jitter by choosing a random INPUT_WIDTH x INPUT_HEIGHT sub-image to process.
	local W, H = options.src:size(3), options.src:size(2)
	local max_ox = math.max(W - config.INPUT_WIDTH, 1)
	local max_oy = math.max(H - config.INPUT_HEIGHT, 1)
	-- ensure that on average, we cover the whole image
	local numJitterIter = math.ceil(W / config.INPUT_WIDTH) * math.ceil(H / config.INPUT_HEIGHT)
	
	-- buffer to store the sub-image deltas prior to adding them all at the end
	local dImg = torch.Tensor(1, unpack(options.src:size():totable())):zero()
	for i = 1, numJitterIter do
		local ox = math.random(1, max_ox)
		local oy = math.random(1, max_oy)
		local ex = math.min(ox + config.INPUT_WIDTH - 1, W)
		local ey = math.min(oy + config.INPUT_HEIGHT - 1, H)
		-- Create a view of a single-image batch
		local batch_view = options.src:view(1, unpack(options.src:size():totable()))
		-- Get the INPUT_WIDTH x INPUT_HEIGHT sub-image
		local src = batch_view[{{},{},{oy, ey}, {ox, ex}}]
		local dsrc = dImg[{{}, {}, {oy, ey}, {ox, ex}}]
		-- Disable any dropout for the forward pass
		net:evaluate()
		local output = net:forward(src)
		-- Some nets require training to be set to enable calculating the derivatives
		net:training()
		local dL_d_output = options.objective(output)
		local dL_d_src = net:updateGradInput(src, output)
		-- src[rgb][x][y] += step_size * normalize(dL/d(src[x][y]))
		local normalized = dL_d_src:mul(1/torch.abs(dL_d_src):mean())
		normalized:mul(options.step_size)
		
		-- add the normalized dL/dInput to the proper section of the image

		dsrc:add(normalized)

		if clip then
			clamp(src)
		end
	end
	options.src:add(dImg)
	if options.jitter > 0 then
		options.src = unjitter(options.src, jx, jy)
	end
	return options.src
end

-- Pick the configured make_step implementation
function make_step(...)
	if config.use_whole_image then
		return make_step_whole_image(...)
	else
		return make_step_chunked(...)
	end
end

-- An awful scaling algorithm without any smoothing
function badscale(img, W, H)
	local w, h = img:size(3), img:size(2)
	local ret = torch.Tensor(3, H, W)
	local xScale = w/W
	local yScale = h/H
	for y = 1, H do
		local sy = math.floor(y * yScale)
		if sy < 1 then sy = 1 end
		if sy > h then sy = h end
		for x = 1, W do
			local sx = math.floor(x * xScale)
			if sx < 1 then sx = 1 end
			if sx > w then sx = w end
			for k = 1, 3 do
				ret[k][y][x] = img[k][sy][sx]
			end
		end
	end
	return ret
end

--[[*
	Run the deepdream process.
	@tparam table options named arguments
	@tparam nn.Sequential net the network to run
	@tparam Tensor[3][W][H] base_image the image to dream
	@tparam number options.iter_n the number of iterations per octave. Default 10
	@tparam number options.octave_n the number of octaves. Default 4
	@tparam number options.octave_scale the amount to scale each octave. Default 1.4
	@tparam boolean options.clip if true, clamp the output to the valid values
	@tparam function options.objective the objective function to use. Default = objective_L2
--]]
function deepdream(options)
	assert(options.net and options.base_image, "net and base_image are required")
	local defaults = {
		iter_n = 10,
		octave_n = 4,
		octave_scale = 1.4,
		end_layer = "inception_4c",
		clip = true
	}
	for k, v in pairs(defaults) do
		if options[k] == nil then options[k] = v end
	end
	assert(options.base_image, "Must provide options.base_image")
	assert(options.net, "Must provide options.net")
	local octaves = { preprocess(options.base_image) }
	for i = 2, options.octave_n do
		local prev = octaves[i - 1]
		local new_width = prev:size(3) / options.octave_scale
		local new_height = prev:size(2) / options.octave_scale
		-- FIXME: This seems to wash out virtually all of the progress; compare with nd.zoom
		local scaled = image.scale(prev, new_width, new_height, 'bilinear')
		octaves[i] = scaled
	end
	local src
	-- TODO: is there a 'zeros_like'?
	local detail = torch.Tensor(unpack(octaves[#octaves]:size():totable())):zero()
	local window
	for octave = #octaves, 1, -1 do
		local octave_base = octaves[octave]
		local h, w = octave_base:size(3), octave_base:size(2)
		if octave > 0 then
			local h1, w1 = detail:size(3), detail:size(2)
			--detail = badscale(detail, w, h)
			detail = image.scale(detail, w, h, 'bilinear')
		end
		-- clone because we need to keep the original to extract the detail at the end
		src = octave_base:clone():add(detail)
		for i = 1, options.iter_n do
			options.src = src
			src = make_step(options)
			config.vis_callback({
				img = src,
				i = i,
				octave = octave
			})
			print(octave, i, options.end_layer, unpack(src:size():totable()))
		end
		detail = src - octave_base
	end
	return deprocess(src)
end
