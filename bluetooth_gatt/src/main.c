/*
 * Copyright (c) 2023 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/gap.h>
#include <zephyr/bluetooth/uuid.h>
#include <zephyr/bluetooth/conn.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <zephyr/sys/printk.h>
#include <zephyr/sys/util.h>
#include <zephyr/drivers/gpio.h>
#include <dk_buttons_and_leds.h>
#include "my_lbs.h"
#include <zephyr/sys/printk.h>
#include "adc.h"

static struct bt_le_adv_param *adv_param = BT_LE_ADV_PARAM(
	(BT_LE_ADV_OPT_CONNECTABLE |
	 BT_LE_ADV_OPT_USE_IDENTITY), /* Connectable advertising and use identity address */
	800,						  /* Min Advertising Interval 500ms (800*0.625ms) */
	801,						  /* Max Advertising Interval 500.625ms (801*0.625ms) */
	NULL);						  /* Set to NULL for undirected advertising */

LOG_MODULE_REGISTER(Lesson4_Exercise2, LOG_LEVEL_INF);

#define DEVICE_NAME CONFIG_BT_DEVICE_NAME
#define DEVICE_NAME_LEN (sizeof(DEVICE_NAME) - 1)

#define RUN_STATUS_LED DK_LED1
#define CON_STATUS_LED DK_LED2
#define USER_LED DK_LED3
#define USER_BUTTON DK_BTN1_MSK

#define STACKSIZE 1024
#define PRIORITY 7

#define RUN_LED_BLINK_INTERVAL 1000
/* STEP 17 - Define the interval at which you want to send data at */
#define NOTIFY_INTERVAL 500
static bool app_button_state;
static int suunta = 0;
/* STEP 15 - Define the data you want to stream over Bluetooth LE */
static uint32_t app_sensor_value = 0;

static const struct bt_data ad[] = {
	BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
	BT_DATA(BT_DATA_NAME_COMPLETE, DEVICE_NAME, DEVICE_NAME_LEN),

};

static const struct bt_data sd[] = {
	BT_DATA_BYTES(BT_DATA_UUID128_ALL, BT_UUID_LBS_VAL),
};

static const struct gpio_dt_spec button = GPIO_DT_SPEC_GET(DT_NODELABEL(button0), gpios);

/* STEP 16 - Define a function to simulate the data */
static void simulate_data(void)
{
    if (initializeADC() != 0)
    {
        printk("ADC initialization failed!");
        return;
    }

}

static void app_led_cb(bool led_state)
{
	dk_set_led(USER_LED, led_state);
}

static bool app_button_cb(void)
{
	return app_button_state;
}

/* STEP 18.1 - Define the thread function */
void send_data_thread(void)
{
    while (1)
    {
        struct Measurement m = readADCValue();
        int val = gpio_pin_get_dt(&button); // Getting the state of the button
        if (val < 0) {
			printk("suunta = %d\n", suunta);
            printk("Error reading button state: %d\n", val);
            // Handle error or continue
        }

        if (val > 0) { // If button is pressed (assuming high state is 'pressed')
            if (suunta < 5) {
				printk("suunta = %d\n", suunta);
                suunta++;
            } else {
				printk("suunta = %d\n", suunta);
                suunta = 1;
            }
        }

        LOG_DBG("x = %d,  y = %d,  z = %d\n", m.x, m.y, m.z);

        // Deconstruct the struct and send the values one at a time
        for (int i = 0; i < 4; i++) {
            if (i == 0) {
                app_sensor_value = m.x;
                my_lbs_send_sensor_notify(app_sensor_value);
                printk("x = %d\n", m.x);
            }
            else if (i == 1) {
                app_sensor_value = m.y;
                my_lbs_send_sensor_notify(app_sensor_value);
                printk("y = %d\n", m.y);
            }
            else if (i == 2) {
                app_sensor_value = m.z;
                my_lbs_send_sensor_notify(app_sensor_value);
                printk("z = %d\n", m.z);
            }
            else if (i == 3) {
                app_sensor_value = suunta;
                my_lbs_send_sensor_notify(app_sensor_value);
                printk("suunta = %d\n", suunta);
            }
        }

        k_sleep(K_MSEC(NOTIFY_INTERVAL));
    }
}

static struct my_lbs_cb app_callbacks = {
	.led_cb = app_led_cb,
	.button_cb = app_button_cb,
};

static void button_changed(uint32_t button_state, uint32_t has_changed)
{
	if (has_changed & USER_BUTTON)
	{
		uint32_t user_button_state = button_state & USER_BUTTON;
		/* STEP 6 - Send indication on a button press */
		my_lbs_send_button_state_indicate(user_button_state);
		app_button_state = user_button_state ? true : false;
	}
}
static void on_connected(struct bt_conn *conn, uint8_t err)
{
	if (err)
	{
		printk("Connection failed (err %u)\n", err);
		return;
	}

	printk("Connected\n");

	dk_set_led_on(CON_STATUS_LED);
}

static void on_disconnected(struct bt_conn *conn, uint8_t reason)
{
	printk("Disconnected (reason %u)\n", reason);

	dk_set_led_off(CON_STATUS_LED);
}

struct bt_conn_cb connection_callbacks = {
	.connected = on_connected,
	.disconnected = on_disconnected,
};

static int init_button(void)
{
    int err;

    if (!device_is_ready(button.port)) {
        printk("Error: button device %s is not ready\n", button.port->name);
        return -ENODEV;
    }

    err = gpio_pin_configure_dt(&button, GPIO_INPUT);
    if (err) {
        printk("Error %d: failed to configure %s pin %d\n", 
               err, button.port->name, button.pin);
        return err;
    }

	return err;
}

void main(void)
{
	int blink_status = 0;
	int err;

	LOG_INF("Starting Lesson 4 - Exercise 2 \n");

	err = dk_leds_init();
	if (err)
	{
		LOG_ERR("LEDs init failed (err %d)\n", err);
		return;
	}

	err = init_button();
	if (err)
	{
		printk("Button init failed (err %d)\n", err);
		return;
	}

	err = bt_enable(NULL);
	if (err)
	{
		LOG_ERR("Bluetooth init failed (err %d)\n", err);
		return;
	}
	bt_conn_cb_register(&connection_callbacks);

	err = my_lbs_init(&app_callbacks);
	if (err)
	{
		printk("Failed to init LBS (err:%d)\n", err);
		return;
	}
	LOG_INF("Bluetooth initialized\n");
	err = bt_le_adv_start(adv_param, ad, ARRAY_SIZE(ad), sd, ARRAY_SIZE(sd));
	if (err)
	{
		LOG_ERR("Advertising failed to start (err %d)\n", err);
		return;
	}

	LOG_INF("Advertising successfully started\n");
	for (;;)
	{
		dk_set_led(RUN_STATUS_LED, (++blink_status) % 2);
		k_sleep(K_MSEC(RUN_LED_BLINK_INTERVAL));

		if (initializeADC() != 0)
		{
			printk("ADC initialization failed!");
			return;
		}

		struct Measurement m = readADCValue();
		LOG_INF("x = %d,  y = %d,  z = %d\n", m.x, m.y, m.z);

		k_sleep(K_MSEC(1000));
	}
}
/* STEP 18.2 - Define and initialize a thread to send data periodically */
K_THREAD_DEFINE(send_data_thread_id, STACKSIZE, send_data_thread, NULL, NULL, NULL, PRIORITY, 0, 0);
