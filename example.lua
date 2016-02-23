require("torch")
require("image")
require("dream")

function printf(fmt, ...)
	print(fmt:format(...))
end

local config = dream.config

net = torch.load(config.model_path)

local src_image = arg[1] or config.input_file

printf("Loading %s...", src_image)

img = image.load(src_image)

local W, H = img:size(3), img:size(2)

if config.max_image_size and math.max(W, H) > config.max_image_size then
	printf("Resizing %d x %d image...", W, H)
	img = image.scale(img, config.max_image_size)
	--local new_height = max_width / W * H
	--img = image.scale(img, max_width, new_height, 'bilinear')
	W, H = img:size(3), img:size(2)
	printf("Resized to %d x %d", W, H)
end

image.display(img)

out = dream.deepdream({
	net = net,
	base_image = img
})
image.display(out)

image.save(config.output_file, out)

printf("Wrote to %s", config.output_file)